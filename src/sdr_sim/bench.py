"""Modo bancada: o simulador como modem do rádio da bancada (Si4463 no Pi).

Na etapa 1 da bancada RF, o rádio da estação é um Raspberry Pi com um Si4463.
O Si4463 não recebe nem produz IQ: transmite e recebe bytes. Este módulo faz
a conversão, com o mesmo código que o simulador já usa para fingir o satélite:

    grs-uplink ─:5559 IQ─▶ BenchUplinkStation ─ demodula ─▶ quadro ─tx─▶ serviço do rádio ))) Bluepill
    cadeia     ◀─:5556 IQ── emissor fs2 ◀─ modula ◀─ quadro ◀─PUB── serviço do rádio ((( Bluepill

O `satellite.py` não entra: quem responde é o firmware da Bluepill, por RF.
O simulador não gera telemetria própria: o beacon e o `0x10` também vêm do ar.

**A API do serviço do rádio** (local no Pi; `bench_radio_stub.py` a imita):

    REQ/REP  (padrão tcp://127.0.0.1:5600), JSON:
        {"op": "tx", "channel": "vhf"|"uhf", "frame_hex": "..."}
            -> {"ok": true, "started_at": iso, "ended_at": iso}
            -> {"ok": false, "error": "..."}
        {"op": "health"}
            -> {"ok": true, "listening": "vhf"|"uhf", "tx_count": n, "rx_count": n, ...}

    PUB      (padrão tcp://127.0.0.1:5601), um quadro por mensagem:
        [b"vhf"|b"uhf", quadro, JSON {"rssi": int|null, "received_at": iso}]

O quadro é o que o Si4463 escreve depois do syncword: size tag + codeword do
NGHam. Preâmbulo e syncword são do chip.

O rádio só responde ao `tx` depois de a rajada sair inteira, com os horários
reais dela. É com eles que este módulo responde `ack` na :5559 — e o
`uplink_transmissions` passa a gravar o que o rádio fez, não uma estimativa.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import zmq

from sdr_sim import fs2_frames, uplink_rx
from sdr_sim.uplink import UplinkStation, _now_iso

DEFAULT_RADIO_ADDRESS = "tcp://127.0.0.1:5600"
DEFAULT_RADIO_RX_ADDRESS = "tcp://127.0.0.1:5601"
CHANNELS = ("vhf", "uhf")
# Taxa de cada canal no Si4463 (configuração do TTC 2.0): quanto tempo um
# quadro fica no ar, para o prazo do `tx`.
CHANNEL_BAUD = {"vhf": 1200, "uhf": 4800}
# Preâmbulo e syncword que o chip põe antes do quadro (config_fs2.h da bancada).
PREAMBLE_BYTES = 4
SYNC_BYTES = 4
# Folga do prazo do `tx` além do tempo no ar: troca de canal (2,8 ms), SPI,
# escalonamento do Pi. Generosa: estourar o prazo devolve o TC à fila.
TX_MARGIN_S = 3.0
HEALTH_PERIOD_S = 2.0
HEALTH_TIMEOUT_S = 1.0
RECENT = 20


def airtime_s(channel: str, frame_bytes: int) -> float:
    """Tempo no ar de um quadro, com o preâmbulo e o syncword do chip."""
    return (PREAMBLE_BYTES + SYNC_BYTES + frame_bytes) * 8 / CHANNEL_BAUD[channel]


def corrected_bits(raw: bytes, clean: bytes) -> int:
    """Bits em que o quadro recebido difere do recodificado (o que o RS corrigiu)."""
    if len(raw) != len(clean):
        return 8 * max(len(raw), len(clean))
    diff = np.bitwise_xor(np.frombuffer(raw, np.uint8), np.frombuffer(clean, np.uint8))
    return int(np.unpackbits(diff).sum())


def _packet_id(frame: bytes) -> str | None:
    packet = uplink_rx.ngham_decode(frame)
    return f"0x{packet[0]:02X}" if packet else None


class RadioClient:
    """Cliente REQ da API do rádio. Um socket por pedido: um REQ que estourou o
    prazo fica preso esperando a resposta, e recriá-lo é o jeito simples de
    seguir (o "lazy pirate" do guia do ZMQ)."""

    def __init__(self, address: str, context: zmq.Context | None = None) -> None:
        self.address = address
        self._context = context or zmq.Context.instance()
        self._lock = threading.Lock()

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
        """A resposta do rádio, ou None se ela não chegou no prazo."""
        with self._lock:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            try:
                socket.connect(self.address)
                socket.send(json.dumps(message).encode())
                if not socket.poll(int(timeout_s * 1000)):
                    return None
                return json.loads(socket.recv())
            finally:
                socket.close(linger=0)

    def tx(self, channel: str, frame: bytes, timeout_s: float) -> dict[str, Any]:
        reply = self.request({"op": "tx", "channel": channel, "frame_hex": frame.hex()}, timeout_s)
        if reply is None:
            return {"ok": False, "timeout": True,
                    "error": f"o rádio não respondeu ao tx em {timeout_s:.1f} s"}
        if not isinstance(reply, dict):
            return {"ok": False, "error": f"resposta ilegível do rádio: {reply!r}"}
        return reply

    def health(self, timeout_s: float = HEALTH_TIMEOUT_S) -> dict[str, Any] | None:
        return self.request({"op": "health"}, timeout_s)


class RadioMonitor(threading.Thread):
    """Pergunta ao rádio se está de pé, a cada `period_s`.

    Serve a dois fins: o painel mostra o estado, e a :5559 recusa na hora,
    com o motivo, quando o rádio está fora — em vez de o TC esperar o prazo
    inteiro para descobrir.
    """

    def __init__(self, client: RadioClient, shutdown: threading.Event,
                 period_s: float = HEALTH_PERIOD_S) -> None:
        super().__init__(daemon=True, name="radio-monitor")
        self._client = client
        self._shutdown = shutdown
        self._period_s = period_s
        self._lock = threading.Lock()
        self.online: bool | None = None      # None: ainda não perguntou
        self.health: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_ok_at: float | None = None
        self.checked_at: float | None = None

    def poll_once(self) -> None:
        try:
            reply = self._client.health()
            error = None if reply and reply.get("ok") else (
                "sem resposta ao health" if reply is None else f"health: {reply.get('error')}")
        except (zmq.ZMQError, ValueError) as exc:
            reply, error = None, f"health: {exc}"
        now = time.time()
        with self._lock:
            self.checked_at = now
            if error is None:
                if self.online is not True:
                    print(f"[sdr-sim] rádio da bancada de pé ({self._client.address})", flush=True)
                self.online, self.health, self.last_error, self.last_ok_at = True, reply, None, now
            else:
                if self.online is not False:
                    print(f"[sdr-sim] rádio da bancada FORA: {error}", flush=True)
                self.online, self.last_error = False, error

    def offline_reason(self) -> str | None:
        """O motivo, se a última pergunta falhou; None se está de pé ou ainda não se sabe."""
        with self._lock:
            if self.online is not False:
                return None
            since = "" if self.last_ok_at is None else f" (última resposta há {time.time() - self.last_ok_at:.0f} s)"
            return f"{self.last_error}{since}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "address": self._client.address,
                "online": self.online,
                "last_error": self.last_error,
                "last_ok_at": self.last_ok_at,
                "checked_at": self.checked_at,
                "health": self.health,
            }

    def run(self) -> None:
        while not self._shutdown.is_set():
            self.poll_once()
            self._shutdown.wait(self._period_s)


class BenchUplinkStation(UplinkStation):
    """A :5559 no modo bancada: demodula a rajada e manda o quadro ao rádio.

    Mesmo contrato do USRP e do satélite simulado. A diferença é a ordem: lá
    o `ack` sai depois da duração da rajada e o satélite processa depois; aqui
    o `ack` só sai depois de o RÁDIO confirmar que transmitiu, com os
    horários dele. Sem confirmação, `timeout` ou `error`, como o USRP.
    """

    def __init__(self, controller, nominal_hz: float, baud: int, radio: RadioClient,
                 shutdown: threading.Event, monitor: RadioMonitor | None = None,
                 channel: str = "vhf") -> None:
        super().__init__(satellite=None, controller=controller, nominal_hz=nominal_hz, baud=baud,
                         respond=lambda radio_name, frame: None, shutdown=shutdown)
        if channel not in CHANNELS:
            raise ValueError(f"canal {channel!r}: esperava vhf ou uhf")
        self.radio = radio
        self.monitor = monitor
        self.channel = channel
        self.frames_sent = 0

    def handle_burst(self, header: dict[str, Any], iq_bytes: bytes):
        burst_id = str(header.get("burst_id"))
        reply, iq, sample_rate, tx_freq, refusal = self._parse_burst(header, iq_bytes)
        if refusal is not None:
            return refusal, None

        reason = self.monitor.offline_reason() if self.monitor is not None else None
        if reason:
            detail = f"rádio da bancada fora do ar: {reason}"
            self._record(burst_id, tx_freq, detail, [])
            return reply("error", detail), None

        try:
            frames = uplink_rx.receive_frames(iq, sample_rate, self.baud)
        except ValueError as error:
            self._record(burst_id, tx_freq, str(error), [])
            return reply("error", str(error)), None
        if not frames:
            detail = "o receptor de uplink não decodificou a rajada; nada foi ao ar"
            self._record(burst_id, tx_freq, detail, [])
            return reply("error", detail), None

        started = ended = None
        corrected = 0
        sizes = []
        for packet, raw in frames:
            # Vai ao ar o quadro RECODIFICADO do pacote, não o cru. O cru traz o
            # que o Reed-Solomon corrigiu, e isso inclui artefato deste
            # receptor: a rajada acaba no último símbolo, e o filtro gaussiano
            # deixa o último bit sem cauda para decidir. Repassar o cru poria
            # no ar um erro que a estação não cometeu. A forma de onda continua
            # conferida (o pacote teve de decodificar), e os bits corrigidos
            # ficam registrados.
            frame = fs2_frames.ngham_frame(packet)
            corrected += corrected_bits(raw, frame)
            sizes.append(len(frame))
            timeout = airtime_s(self.channel, len(frame)) + TX_MARGIN_S
            result = self.radio.tx(self.channel, frame, timeout)
            if not result.get("ok"):
                status = "timeout" if result.get("timeout") else "error"
                detail = f"rádio da bancada: {result.get('error') or 'falha sem motivo'}"
                self._record(burst_id, tx_freq, detail, [p for p, _ in frames])
                return reply(status, detail), None
            started = started or result.get("started_at")
            ended = result.get("ended_at")
            self.frames_sent += 1

        detail = (f"bancada: {len(frames)} quadro(s) pelo rádio do Pi, canal {self.channel}"
                  f" ({', '.join(f'{n} B' for n in sizes)}); o receptor de uplink corrigiu "
                  f"{corrected} bit(s)")
        self._record(burst_id, tx_freq, f"transmitido pelo rádio ({corrected} bit(s) corrigido(s) "
                     "na recepção)", [p for p, _ in frames], started, ended)
        return reply("ack", detail, started_at=started or _now_iso(), ended_at=ended or _now_iso(),
                     actual_freq_hz=tx_freq), None

    def _record(self, burst_id: str, tx_freq: float, outcome: str, packets: list[bytes],
                started: str | None = None, ended: str | None = None) -> None:
        with self._lock:
            self.bursts += 1
            self._recent.appendleft({
                "at": round(time.time(), 1),
                "burst_id": burst_id,
                "tx_freq_hz": tx_freq,
                "residual_hz": None,
                "outcome": outcome,
                "packets": [f"0x{p[0]:02X} ({len(p)} B)" for p in packets],
                "responses": [],
                "radio_started_at": started,
                "radio_ended_at": ended,
            })
        print(f"[sdr-sim] uplink {burst_id} (bancada): {outcome}", flush=True)

    def apply(self, changes: dict[str, Any]) -> list[Callable[[], None]]:
        if not isinstance(changes, dict):
            raise ValueError("uplink precisa ser um objeto")
        if {"fail_next", "drop_next"} & set(changes):
            raise ValueError("no modo bancada quem injeta falha é a Bluepill (console serial: "
                             "fail_next, drop_next)")
        return super().apply(changes)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "bancada",
                "channel": self.channel,
                "tx_enabled": self.tx_enabled,
                "nominal_hz": self.nominal_hz,
                "bursts": self.bursts,
                "frames_sent": self.frames_sent,
                "recent_bursts": list(self._recent),
            }

    def last_view(self) -> dict[str, Any] | None:
        return None


class BenchReceiver(threading.Thread):
    """SUB dos quadros que o rádio recebe: os deste canal entram na fila do emissor fs2.

    A modulação (`enqueue`) roda aqui, fora do laço de geração: no Pi ela leva
    ~0,75 s para um `0x10` a 1200 baud, e dentro do laço abriria um buraco no IQ.
    """

    def __init__(self, address: str, channel: str, enqueue: Callable[[bytes], None],
                 shutdown: threading.Event) -> None:
        super().__init__(daemon=True, name="bench-receiver")
        if channel not in CHANNELS:
            raise ValueError(f"canal {channel!r}: esperava vhf ou uhf")
        self.address = address
        self.channel = channel
        self._enqueue = enqueue
        self._shutdown = shutdown
        self._lock = threading.Lock()
        self.received = 0
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT)

    def handle(self, parts: list[bytes]) -> bool:
        """Um quadro do rádio. :return: se entrou na fila."""
        if len(parts) < 2 or parts[0] != self.channel.encode() or not parts[1]:
            return False
        frame = parts[1]
        meta: dict[str, Any] = {}
        if len(parts) > 2:
            try:
                meta = json.loads(parts[2])
            except (ValueError, UnicodeDecodeError):
                meta = {}
        self._enqueue(frame)
        with self._lock:
            self.received += 1
            self._recent.appendleft({
                "at": round(time.time(), 1),
                "pkt_id": _packet_id(frame),
                "bytes": len(frame),
                "rssi": meta.get("rssi"),
                "received_at": meta.get("received_at"),
            })
        return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"address": self.address, "channel": self.channel,
                    "received": self.received, "recent": list(self._recent)}

    def run(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self.channel.encode())
        socket.connect(self.address)
        print(f"[sdr-sim] quadros do rádio da bancada ({self.channel}) de {self.address}", flush=True)
        while not self._shutdown.is_set():
            if not socket.poll(500):
                continue
            self.handle(socket.recv_multipart())
        socket.close(linger=0)


def is_ngham_frame(payload: bytes) -> bool:
    """Para o monitor de pacotes: no modo bancada, qualquer quadro NGHam válido é legítimo."""
    return uplink_rx.ngham_decode(payload) is not None
