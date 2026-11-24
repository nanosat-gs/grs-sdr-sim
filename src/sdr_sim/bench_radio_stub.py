"""Rádio falso da bancada: a API do serviço do rádio, respondida pelo `satellite.py`.

Faz o papel do Pi + Si4463 + Bluepill de uma vez, para testar a estação no
modo bancada sem RF e sem o firmware do satélite. A API é a de `bench.py`:

    REQ/REP  tx | health | sat | beacon
    PUB      [b"vhf"|b"uhf", quadro, {"rssi", "received_at", "stub": true}]

O que ele imita:

- `tx`: o quadro fica "no ar" pelo tempo real da rajada (preâmbulo + sync +
  quadro, na taxa do canal) antes da resposta, com os horários de início e fim;
- o satélite: o pacote passa pelo `satellite.py` e a resposta sai no canal que
  ele manda (o pong pela VHF, o `0x13` pela UHF), depois de `response_delay_s`
  e do tempo no ar dela;
- o beacon (`0x00`, VHF) e logo depois o `0x10` (UHF), a cada `beacon_period_s`.

O que ele NÃO imita: a regra de escuta do Pi (um rádio para dois canais), a
perda de quadro, o RSSI real. O beacon é um marcador (ID `0x00` + indicativo),
não o beacon do EPS2.

Uso:
    python -m sdr_sim.bench_radio_stub --keys-file keys.test.json --beacon-period 30
    op extra, só do rádio falso:  {"op": "sat", "fail_next": -2} | {"op": "sat", "drop_next": true}
                                  {"op": "beacon"}  (manda o beacon e o 0x10 agora)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import zmq

from sdr_sim import fs2_frames, uplink_rx
from sdr_sim.bench import (CHANNELS, DEFAULT_RADIO_ADDRESS, DEFAULT_RADIO_RX_ADDRESS,
                           airtime_s)
from sdr_sim.satellite import SatelliteModel, load_keys

EPS_BEACON_ID = 0x00
# O que o satélite leva entre ouvir o TC e começar a responder (processar,
# trocar de canal). Valor de modelo; o da Bluepill será medido.
DEFAULT_RESPONSE_DELAY_S = 0.2
# Entre o fim do beacon e o início do 0x10.
BEACON_TO_TELEMETRY_S = 0.1
STUB_RSSI = 180


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def beacon_packet() -> bytes:
    return bytes([EPS_BEACON_ID]) + fs2_frames.CALLSIGN + b"BANCADA-STUB"


class StubRadio:
    def __init__(self, satellite: SatelliteModel, publish: Callable[[str, bytes, dict], None],
                 response_delay_s: float = DEFAULT_RESPONSE_DELAY_S,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        self.satellite = satellite
        self._publish = publish
        self.response_delay_s = response_delay_s
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self.tx_count = 0
        self.rx_count = 0
        self._telemetry = fs2_frames.GeneralTelemetryStream(clock)
        self.responder: threading.Thread | None = None

    # --- REQ/REP -------------------------------------------------------------------

    def handle(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return {"ok": False, "error": "pedido precisa ser um objeto JSON"}
        op = request.get("op")
        if op == "health":
            with self._lock:
                return {"ok": True, "stub": True, "listening": "vhf",
                        "tx_count": self.tx_count, "rx_count": self.rx_count}
        if op == "tx":
            return self._tx(request)
        if op == "sat":
            return self._sat(request)
        if op == "beacon":
            threading.Thread(target=self.send_beacon, daemon=True).start()
            return {"ok": True}
        return {"ok": False, "error": f"op desconhecida: {op!r}"}

    def _tx(self, request: dict[str, Any]) -> dict[str, Any]:
        channel = request.get("channel")
        if channel not in CHANNELS:
            return {"ok": False, "error": f"canal {channel!r}: esperava vhf ou uhf"}
        try:
            frame = bytes.fromhex(request.get("frame_hex") or "")
        except ValueError:
            return {"ok": False, "error": "frame_hex não é hexadecimal"}
        if not frame:
            return {"ok": False, "error": "quadro vazio"}

        started = self._clock()
        self._sleep(airtime_s(channel, len(frame)))
        ended = self._clock()
        with self._lock:
            self.tx_count += 1
        # Guardada para quem precisa esperar a resposta (os testes).
        self.responder = threading.Thread(target=self._satellite_hears, args=(channel, frame),
                                          daemon=True)
        self.responder.start()
        return {"ok": True, "started_at": _iso(started), "ended_at": _iso(ended)}

    def _sat(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            if "fail_next" in request:
                self.satellite.fail_next(int(request["fail_next"]))
            if request.get("drop_next") is True:
                self.satellite.drop_next()
        except (TypeError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        return {"ok": True}

    # --- o "satélite" ----------------------------------------------------------------

    def _satellite_hears(self, channel: str, frame: bytes) -> None:
        packet = uplink_rx.ngham_decode(frame)
        if packet is None:
            print(f"[radio-stub] quadro de {len(frame)} B pela {channel.upper()}: NGHam não "
                  "decodificou", flush=True)
            return
        responses = self.satellite.process(packet)
        print(f"[radio-stub] TC 0x{packet[0]:02X} pela {channel.upper()}: "
              f"{len(responses)} resposta(s)", flush=True)
        for response in responses:
            self._sleep(self.response_delay_s)
            self.receive(response.radio, fs2_frames.ngham_frame(response.packet))

    def receive(self, channel: str, frame: bytes) -> None:
        """O rádio da estação "recebe" um quadro: espera o tempo no ar e publica."""
        self._sleep(airtime_s(channel, len(frame)))
        with self._lock:
            self.rx_count += 1
        self._publish(channel, frame, {"rssi": STUB_RSSI, "received_at": _iso(self._clock()),
                                       "stub": True})

    def send_beacon(self) -> None:
        self.receive("vhf", fs2_frames.ngham_frame(beacon_packet()))
        self._sleep(BEACON_TO_TELEMETRY_S)
        self.receive("uhf", self._telemetry())


class Publisher:
    def __init__(self, socket: zmq.Socket) -> None:
        self._socket = socket
        self._lock = threading.Lock()

    def __call__(self, channel: str, frame: bytes, meta: dict) -> None:
        with self._lock:
            self._socket.send_multipart([channel.encode(), frame, json.dumps(meta).encode()])


_shutdown = threading.Event()


def _handle_signal(_signum, _frame) -> None:
    _shutdown.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bind", default=DEFAULT_RADIO_ADDRESS.replace("127.0.0.1", "*"),
                        help="REP da API (padrão tcp://*:5600)")
    parser.add_argument("--pub-bind", default=DEFAULT_RADIO_RX_ADDRESS.replace("127.0.0.1", "*"),
                        help="PUB dos quadros recebidos (padrão tcp://*:5601)")
    parser.add_argument("--keys-file", default=None, help="JSON de chaves de TESTE do grs-uplink")
    parser.add_argument("--beacon-period", type=float, default=30.0,
                        help="segundos entre beacons (0 desliga)")
    parser.add_argument("--response-delay", type=float, default=DEFAULT_RESPONSE_DELAY_S)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    context = zmq.Context()
    rep = context.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(args.bind)
    pub = context.socket(zmq.PUB)
    pub.bind(args.pub_bind)

    radio = StubRadio(SatelliteModel(load_keys(args.keys_file)), Publisher(pub),
                      response_delay_s=args.response_delay)
    print(f"[radio-stub] rádio FALSO da bancada: API em {args.bind}, quadros em {args.pub_bind}",
          flush=True)

    def beacons() -> None:
        while not _shutdown.wait(args.beacon_period):
            radio.send_beacon()

    if args.beacon_period > 0:
        threading.Thread(target=beacons, daemon=True, name="beacon").start()
        print(f"[radio-stub] beacon + 0x10 a cada {args.beacon_period:.0f} s", flush=True)

    while not _shutdown.is_set():
        if not rep.poll(500):
            continue
        try:
            reply = radio.handle(json.loads(rep.recv()))
        except (ValueError, UnicodeDecodeError) as error:
            reply = {"ok": False, "error": f"pedido ilegível: {error}"}
        rep.send(json.dumps(reply).encode())

    rep.close(linger=0)
    pub.close(linger=0)
    context.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
