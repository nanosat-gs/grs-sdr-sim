"""Gateway da bancada: o rádio de rede (o Pi) com o nome do rádio da estação.

Na etapa 1 da bancada RF, o rádio da estação é um Raspberry Pi que roda este
mesmo simulador em modo bancada (`bench.py`). Os consumidores da estação
(demodulador, FFT, gravador, `grs-uplink`) procuram o rádio pelo nome e pelo
IP fixo (`grs-iq-rx`, `grs-iq-rx-uhf`). O gateway fica com esse nome e repassa
tudo ao Pi, sem processar nada:

    PUB  --iq-bind   (*:5556) ◀── SUB --iq-source   (Pi:5556 ou Pi:5566)   o IQ da cadeia
    REP  --tx-bind   (*:5559) ──▶ REQ --tx-target   (Pi:5559)              o TX, com prazo
    PUB  --tune-bind (*:5575) ◀── SUB --tune-source (sintetizador:5557)    o tune, para o Pi assinar

O sintetizador só publica dentro da rede do Docker; o Pi assina o `tune` na
porta que o gateway publica no host.

**Pi fora do ar.** O gateway sabe que o Pi está de pé pelo IQ: o simulador
publica sem parar. Sem IQ há mais de `--stale-after` segundos, a :5559
responde `error` NA HORA, com o motivo — o TC volta à fila e o operador vê
por quê, em vez de esperar o prazo inteiro. Com IQ chegando, o pedido segue
para o Pi; sem resposta no prazo, `timeout`.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from typing import Any

import zmq

DEFAULT_IQ_BIND = "tcp://*:5556"
DEFAULT_TX_BIND = "tcp://*:5559"
# Sem IQ do Pi por mais que isto, ele é dado como fora do ar.
DEFAULT_STALE_AFTER_S = 3.0
# Folga do prazo do TX além da duração da rajada: demodular no Pi (65 ms
# medidos), o rádio transmitir e confirmar. Menor que a do grs-uplink
# (UPLINK_RADIO_TIMEOUT_S, 10 s), para o motivo do gateway chegar antes do
# prazo dele.
DEFAULT_TX_MARGIN_S = 7.0


class IqRelay(threading.Thread):
    """SUB do IQ do Pi -> PUB da cadeia. Conta e marca a hora da última mensagem."""

    def __init__(self, source: str, bind: str, shutdown: threading.Event,
                 context: zmq.Context) -> None:
        super().__init__(daemon=True, name="iq-relay")
        self.source = source
        self._bind = bind
        self._shutdown = shutdown
        self._context = context
        self._lock = threading.Lock()
        self.messages = 0
        self.last_at: float | None = None
        self._was_alive: bool | None = None

    def silence_s(self) -> float | None:
        """Segundos desde o último IQ; None se nunca chegou."""
        with self._lock:
            return None if self.last_at is None else time.monotonic() - self.last_at

    def run(self) -> None:
        sub = self._context.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.RCVHWM, 10_000)
        sub.connect(self.source)
        pub = self._context.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 10_000)
        pub.bind(self._bind)
        print(f"[gateway] IQ de {self.source} -> {self._bind}", flush=True)
        while not self._shutdown.is_set():
            if not sub.poll(500):
                continue
            parts = sub.recv_multipart(copy=False)
            pub.send_multipart(parts, copy=False)
            with self._lock:
                self.messages += 1
                self.last_at = time.monotonic()
        sub.close(linger=0)
        pub.close(linger=0)


class TuneRelay(threading.Thread):
    """SUB do tune do sintetizador -> PUB que o Pi assina, mensagem inteira (multipart)."""

    def __init__(self, source: str, bind: str, shutdown: threading.Event,
                 context: zmq.Context) -> None:
        super().__init__(daemon=True, name="tune-relay")
        self._source = source
        self._bind = bind
        self._shutdown = shutdown
        self._context = context
        self.forwarded = 0

    def run(self) -> None:
        sub = self._context.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.connect(self._source)
        pub = self._context.socket(zmq.PUB)
        pub.bind(self._bind)
        print(f"[gateway] tune de {self._source} -> {self._bind}", flush=True)
        while not self._shutdown.is_set():
            if not sub.poll(500):
                continue
            pub.send_multipart(sub.recv_multipart())
            self.forwarded += 1
        sub.close(linger=0)
        pub.close(linger=0)


class TxRelay:
    """A :5559 da estação, repassada ao Pi. Um pedido por vez, como um rádio."""

    def __init__(self, target: str, iq: IqRelay, context: zmq.Context,
                 stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 margin_s: float = DEFAULT_TX_MARGIN_S) -> None:
        self.target = target
        self._iq = iq
        self._context = context
        self.stale_after_s = stale_after_s
        self.margin_s = margin_s

    def handle(self, parts: list[bytes]) -> bytes:
        try:
            header: dict[str, Any] = json.loads(parts[0])
        except (ValueError, UnicodeDecodeError, IndexError) as error:
            return json.dumps({"burst_id": None, "status": "error",
                               "detail": f"pedido ilegível: {error}"}).encode()
        # Texto, como o simulador e o receptor USRP devolvem.
        burst_id = None if header.get("burst_id") is None else str(header["burst_id"])

        def reply(status: str, detail: str) -> bytes:
            print(f"[gateway] TX {burst_id}: {status} ({detail})", flush=True)
            return json.dumps({"burst_id": burst_id, "status": status, "detail": detail}).encode()

        silence = self._iq.silence_s()
        if silence is None:
            return reply("error", f"Pi da bancada sem IQ desde que o gateway subiu "
                                  f"({self._iq.source}): desligado ou fora da rede?")
        if silence > self.stale_after_s:
            return reply("error", f"Pi da bancada sem IQ há {silence:.0f} s "
                                  f"({self._iq.source}): desligado ou fora da rede?")

        try:
            duration = float(header["n_samples"]) / float(header["sample_rate"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            duration = len(parts[1]) / 8 / 240_000 if len(parts) > 1 else 0.0
        timeout_s = duration + self.margin_s

        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.target)
            socket.send_multipart(parts)
            if not socket.poll(int(timeout_s * 1000)):
                return reply("timeout", f"o Pi não respondeu à :5559 em {timeout_s:.1f} s "
                                        f"({self.target})")
            answer = socket.recv()
        finally:
            socket.close(linger=0)
        try:
            status = json.loads(answer).get("status")
        except (ValueError, UnicodeDecodeError, AttributeError):
            status = "?"
        print(f"[gateway] TX {burst_id}: {status} (do Pi)", flush=True)
        return answer


_shutdown = threading.Event()


def _handle_signal(_signum, _frame) -> None:
    _shutdown.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iq-source", required=True, help="IQ do Pi, ex.: tcp://172.16.9.226:5556")
    parser.add_argument("--iq-bind", default=DEFAULT_IQ_BIND)
    parser.add_argument("--tx-target", default=None, help="a :5559 do Pi (só na cadeia VHF)")
    parser.add_argument("--tx-bind", default=DEFAULT_TX_BIND)
    parser.add_argument("--tune-source", default=None, help="o sintetizador desta cadeia")
    parser.add_argument("--tune-bind", default=None, help="onde o Pi assina o tune, ex.: tcp://*:5575")
    parser.add_argument("--stale-after", type=float, default=DEFAULT_STALE_AFTER_S)
    parser.add_argument("--tx-margin", type=float, default=DEFAULT_TX_MARGIN_S)
    args = parser.parse_args(argv)
    if bool(args.tune_source) != bool(args.tune_bind):
        parser.error("--tune-source e --tune-bind andam juntos")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    context = zmq.Context()
    iq = IqRelay(args.iq_source, args.iq_bind, _shutdown, context)
    iq.start()
    if args.tune_source:
        TuneRelay(args.tune_source, args.tune_bind, _shutdown, context).start()

    rep = None
    tx = None
    if args.tx_target:
        tx = TxRelay(args.tx_target, iq, context, args.stale_after, args.tx_margin)
        rep = context.socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        rep.bind(args.tx_bind)
        print(f"[gateway] TX {args.tx_bind} -> {args.tx_target}", flush=True)

    alive: bool | None = None
    while not _shutdown.is_set():
        if rep is not None and rep.poll(500):
            rep.send(tx.handle(rep.recv_multipart()))
        elif rep is None:
            _shutdown.wait(0.5)
        silence = iq.silence_s()
        now_alive = silence is not None and silence <= args.stale_after
        if now_alive != alive:
            print(f"[gateway] Pi {'de pé: IQ chegando' if now_alive else 'SEM IQ'} "
                  f"({args.iq_source})", flush=True)
            alive = now_alive

    if rep is not None:
        rep.close(linger=0)
    iq.join(timeout=2.0)
    context.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
