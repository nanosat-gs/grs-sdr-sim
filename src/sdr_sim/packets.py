"""Escuta a ponta final do cano e confere o que chegou contra o que foi enviado.

O simulador é o único componente que sabe exatamente o que transmitiu. Assinar
os raw packets do detector (:5558) fecha a malha dentro do próprio painel:
mude o SNR e veja, no mesmo lugar, quantos pacotes deixam de chegar ou chegam
corrompidos.

Só OBSERVA. Nada no caminho de dados depende disto; se o detector não estiver
de pé, o painel mostra "sem pacotes" e o simulador segue transmitindo.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any

import zmq

RECENT_PACKETS = 15
RATE_WINDOW_S = 10.0


class PacketMonitor(threading.Thread):
    def __init__(self, address: str, expected_payload: bytes, shutdown: threading.Event) -> None:
        super().__init__(daemon=True, name="packet-monitor")
        self.address = address
        self._expected = expected_payload
        self._shutdown = shutdown
        self._lock = threading.Lock()

        self._total = 0
        self._ok = 0
        self._diverge = 0
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT_PACKETS)
        self._arrivals: deque[float] = deque()
        # NÃO `_started`: threading.Thread já usa esse nome para um Event
        # interno, e sobrescrevê-lo quebra o .start() com um AttributeError.
        self._counting_since = time.monotonic()

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVTIMEO, 500)
        socket.connect(self.address)
        print(f"[sdr-sim] conferindo pacotes em {self.address}", flush=True)

        while not self._shutdown.is_set():
            try:
                frames = socket.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            self.record(frames)

        socket.close(linger=0)
        context.term()

    def record(self, frames: list[bytes]) -> None:
        """Um raw packet: [tópico][cabeçalho JSON][payload]."""
        if len(frames) != 3:
            return

        try:
            header = json.loads(frames[1])
        except (ValueError, UnicodeDecodeError):
            return

        payload = frames[2]
        ok = payload[: len(self._expected)] == self._expected
        now = time.monotonic()

        with self._lock:
            self._total += 1
            if ok:
                self._ok += 1
            else:
                self._diverge += 1
            self._arrivals.append(now)
            self._recent.appendleft({
                "seq": header.get("seq"),
                "bit_offset": header.get("bit_offset"),
                "ok": ok,
                "prefix": payload[:8].hex(),
                "age_s": now,
            })

    def reset(self) -> None:
        with self._lock:
            self._total = self._ok = self._diverge = 0
            self._recent.clear()
            self._arrivals.clear()
            self._counting_since = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()

        with self._lock:
            while self._arrivals and now - self._arrivals[0] > RATE_WINDOW_S:
                self._arrivals.popleft()

            # Janela menor que 10 s logo após o início/zerar: dividir por 10
            # subestimaria a taxa nos primeiros segundos.
            window = min(RATE_WINDOW_S, max(now - self._counting_since, 1e-3))

            return {
                "source": self.address,
                "total": self._total,
                "ok": self._ok,
                "diverge": self._diverge,
                "rate_per_s": len(self._arrivals) / window,
                "recent": [
                    {**packet, "age_s": round(now - packet["age_s"], 1)}
                    for packet in self._recent
                ],
            }
