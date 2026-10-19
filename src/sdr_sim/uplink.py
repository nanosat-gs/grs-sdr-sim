"""O satélite simulado do lado do uplink: o "SDR de TX" da :5559 e o caminho das respostas.

    grs-uplink ──REQ :5559 [header, IQ]──▶ TxServer ──▶ UplinkStation.handle_burst
                                                          │ espera a duração real da rajada, responde ack
                                                          │ o satélite ouve? (horizonte, Doppler residual)
                                                          │ uplink_rx.receive -> SatelliteModel.process
                                                          ▼
                       resposta pela VHF: fila do emissor fs2 DESTE simulador
                       resposta pela UHF: PUB :5567 tópico "uhf" -> ResponseListener do sim UHF

É o mesmo contrato da :5559 que o `grs-iq-rx/usrp` implementa com o USRP
(grs-station/docs/tx-datapath.md). O `grs-uplink` não sabe qual dos dois
está do outro lado — que é o ponto, como no RX.

**O satélite só ouve o que chega na frequência dele.** Com a órbita ligada, a
rajada que sai em `tx_freq` chega a bordo em `tx_freq + Doppler`. Se o
residual passar de `tolerance_hz`, o rádio de bordo não ouve: é assim que um
erro de SINAL na correção de Doppler do uplink aparece aqui, em vez de passar
despercebido. A tolerância é um modelo, não uma medida do Si446x.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import zmq

from sdr_sim import fs2_frames, uplink_rx
from sdr_sim.satellite import SatelliteModel

DEFAULT_TOLERANCE_HZ = 1500.0
MAX_BURST_SECONDS = 10.0
RECENT = 20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UplinkStation:
    def __init__(self, satellite: SatelliteModel, controller, nominal_hz: float, baud: int,
                 respond: Callable[[str, bytes], None], shutdown: threading.Event,
                 tolerance_hz: float = DEFAULT_TOLERANCE_HZ, rx_max_bytes: int = 0) -> None:
        self.satellite = satellite
        self._controller = controller
        self.nominal_hz = nominal_hz
        self.baud = baud
        self._respond = respond
        self._shutdown = shutdown
        self.tolerance_hz = tolerance_hz
        self.rx_max_bytes = rx_max_bytes
        self.tx_enabled = True
        self._lock = threading.Lock()
        self.bursts = 0
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT)

    # --- a :5559 -----------------------------------------------------------------

    def handle_burst(self, header: dict[str, Any], iq_bytes: bytes):
        """:return: (resposta ao grs-uplink, o que fazer depois de responder)."""
        burst_id = str(header.get("burst_id"))

        def reply(status: str, detail: str | None = None, **extra) -> dict[str, Any]:
            return {"burst_id": burst_id, "status": status, "detail": detail, **extra}

        try:
            sample_rate = int(header["sample_rate"])
            tx_freq = float(header["tx_freq_hz"])
        except (KeyError, TypeError, ValueError):
            return reply("error", "header sem sample_rate/tx_freq_hz"), None
        if len(iq_bytes) % 8 or not iq_bytes:
            return reply("error", "IQ precisa ser cf32_le (8 bytes por amostra)"), None
        iq = np.frombuffer(iq_bytes, dtype=np.complex64)
        duration = len(iq) / sample_rate
        if duration > MAX_BURST_SECONDS:
            return reply("error", f"rajada de {duration:.1f} s; o limite é {MAX_BURST_SECONDS:.0f} s"), None
        if not self.tx_enabled:
            return reply("tx_disabled", "TX desligado no painel do simulador"), None

        started = _now_iso()
        # O rádio de verdade só confirma depois de a rajada sair inteira.
        self._shutdown.wait(duration)
        ack = reply("ack", started_at=started, ended_at=_now_iso(), actual_freq_hz=tx_freq)

        def deliver() -> None:
            self._deliver(iq, sample_rate, tx_freq, burst_id)

        return ack, deliver

    # --- o satélite ouvindo ------------------------------------------------------------

    def _heard(self, tx_freq: float) -> tuple[bool, str, float | None]:
        fs2 = self._controller.emitter("fs2")
        with self._controller.lock:
            elapsed = self._controller.spectrum.elapsed_s
            if not fs2.audible_at(elapsed):
                return False, "satélite abaixo do horizonte", None
            shift = fs2.doppler.shift_at(elapsed, tx_freq) if fs2.doppler else 0.0
        residual = tx_freq + shift - self.nominal_hz
        if abs(residual) > self.tolerance_hz:
            return False, f"fora da banda do rádio de bordo ({residual:+.0f} Hz)", residual
        return True, "ouvido", residual

    def _deliver(self, iq: np.ndarray, sample_rate: int, tx_freq: float, burst_id: str) -> None:
        heard, why, residual = self._heard(tx_freq)
        packets: list[bytes] = []
        responses = []
        if heard:
            packets = uplink_rx.receive(iq, sample_rate, self.baud, self.rx_max_bytes)
            if not packets:
                why = "ouvido, mas o NGHam não decodificou"
            for packet in packets:
                for response in self.satellite.process(packet):
                    self._respond(response.radio, fs2_frames.ngham_frame(response.packet))
                    responses.append(response)

        with self._lock:
            self.bursts += 1
            self._recent.appendleft({
                "at": round(time.time(), 1),
                "burst_id": burst_id,
                "tx_freq_hz": tx_freq,
                "residual_hz": None if residual is None else round(residual, 1),
                "outcome": why,
                "packets": [f"0x{p[0]:02X} ({len(p)} B)" for p in packets],
                "responses": [f"0x{r.packet[0]:02X} pela {r.radio.upper()}" for r in responses],
            })
        print(f"[sdr-sim] uplink {burst_id}: {tx_freq / 1e6:.6f} MHz, {why}; "
              f"{len(packets)} pacote(s), {len(responses)} resposta(s)", flush=True)

    # --- painel ---------------------------------------------------------------------

    def apply(self, changes: dict[str, Any]) -> list[Callable[[], None]]:
        """Valida tudo e devolve as ações (o SimController aplica todas ou nenhuma)."""
        if not isinstance(changes, dict):
            raise ValueError("uplink precisa ser um objeto")
        unknown = set(changes) - {"tx_enabled", "fail_next", "drop_next"}
        if unknown:
            raise ValueError(f"campo desconhecido em uplink: {', '.join(sorted(unknown))}")
        actions: list[Callable[[], None]] = []
        if "tx_enabled" in changes:
            if not isinstance(changes["tx_enabled"], bool):
                raise ValueError("uplink.tx_enabled precisa ser booleano")
            enabled = changes["tx_enabled"]
            actions.append(lambda: setattr(self, "tx_enabled", enabled))
        if "fail_next" in changes:
            code = changes["fail_next"]
            if isinstance(code, bool) or not isinstance(code, int) or not -32768 <= code <= 32767:
                raise ValueError("uplink.fail_next precisa ser um código int16 (ex.: -2)")
            actions.append(lambda: self.satellite.fail_next(code))
        if changes.get("drop_next") is True:
            actions.append(self.satellite.drop_next)
        return actions

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tx_enabled": self.tx_enabled,
                "nominal_hz": self.nominal_hz,
                "tolerance_hz": self.tolerance_hz,
                "rx_max_bytes": self.rx_max_bytes,
                "bursts": self.bursts,
                "recent_bursts": list(self._recent),
                "satellite": self.satellite.snapshot(),
            }


class TxServer(threading.Thread):
    """REP :5559. Um pedido por vez, como um rádio que só transmite uma rajada de cada vez."""

    def __init__(self, bind: str, station: UplinkStation, shutdown: threading.Event) -> None:
        super().__init__(daemon=True, name="tx-server")
        self._bind = bind
        self._station = station
        self._shutdown = shutdown

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self._bind)
        print(f"[sdr-sim] TX (satélite simulado) em {self._bind}", flush=True)
        while not self._shutdown.is_set():
            if not socket.poll(500):
                continue
            parts = socket.recv_multipart()
            after = None
            try:
                header = json.loads(parts[0])
                reply, after = self._station.handle_burst(header, parts[1] if len(parts) > 1 else b"")
            except (ValueError, UnicodeDecodeError, AttributeError) as error:
                reply = {"burst_id": None, "status": "error", "detail": f"pedido ilegível: {error}"}
            socket.send(json.dumps(reply).encode())
            if after is not None:
                after()
        socket.close(linger=0)
        context.term()


class ResponsePublisher:
    """PUB :5567: [b"vhf"|b"uhf", quadro NGHam depois do syncword]."""

    def __init__(self, bind: str) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(bind)
        self._lock = threading.Lock()

    def publish(self, radio: str, frame: bytes) -> None:
        with self._lock:
            self._socket.send_multipart([radio.encode(), frame])


class ResponseListener(threading.Thread):
    """SUB do PUB de respostas: o que for deste rádio entra na fila do emissor fs2."""

    def __init__(self, address: str, channel: str, enqueue: Callable[[bytes], None],
                 shutdown: threading.Event) -> None:
        super().__init__(daemon=True, name="response-listener")
        self._address = address
        self._topic = channel.encode()
        self._enqueue = enqueue
        self._shutdown = shutdown

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        socket.connect(self._address)
        print(f"[sdr-sim] respostas do satélite ({self._topic.decode()}) de {self._address}", flush=True)
        while not self._shutdown.is_set():
            if not socket.poll(500):
                continue
            parts = socket.recv_multipart()
            if len(parts) == 2 and parts[0] == self._topic:
                self._enqueue(parts[1])
        socket.close(linger=0)
        context.term()


def is_response_frame(data: bytes) -> bool:
    """Para o monitor de pacotes: o quadro é uma resposta do satélite (0x02/0x13)?"""
    packet = uplink_rx.ngham_decode(data)
    return bool(packet) and packet[0] in (0x02, 0x13) and packet[1:8] == fs2_frames.CALLSIGN
