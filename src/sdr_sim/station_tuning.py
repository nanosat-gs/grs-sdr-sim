"""Escuta o que o Station Manager anuncia na :5581 — só para comparar.

O Station Manager publica a portadora (`[freq]`) e o Doppler previsto
(`[doppler]`) do satélite que está rastreando. O simulador NÃO aplica esse
número ao sinal: se aplicasse, o desvio imposto seria exatamente o que a
estação corrige, e o erro sairia zero mesmo com a conta da estação errada.

O que ele faz é pôr os dois lado a lado no painel. O Doppler do simulador vem
de geometria própria (`orbit.py`); o da estação, da spacelab-tracking. Dois
cálculos independentes da mesma órbita que concordam em poucos hertz são a
melhor evidência que dá para ter, sem satélite, de que o sinal e a magnitude
da correção estão certos.
"""

from __future__ import annotations

import threading
import time

import zmq


class StationTuningMonitor(threading.Thread):
    """Assina `[freq]` e `[doppler]` e guarda o último de cada."""

    def __init__(self, address: str, shutdown: threading.Event) -> None:
        super().__init__(daemon=True, name="station-tuning")
        self.address = address
        self._shutdown = shutdown
        self._lock = threading.Lock()
        self._frequency_hz: float | None = None
        self._doppler_hz: float | None = None
        self._doppler_at: float | None = None

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"freq")
        socket.setsockopt(zmq.SUBSCRIBE, b"doppler")
        socket.setsockopt(zmq.RCVTIMEO, 500)
        socket.connect(self.address)
        print(f"[sdr-sim] comparando com o Doppler anunciado em {self.address}", flush=True)

        while not self._shutdown.is_set():
            try:
                frames = socket.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            self.handle(frames)

        socket.close()
        context.term()

    def handle(self, frames: list[bytes]) -> None:
        if len(frames) != 2:
            return
        try:
            value = float(frames[1].decode())
        except (ValueError, UnicodeDecodeError):
            return
        with self._lock:
            if frames[0] == b"freq":
                self._frequency_hz = value
            elif frames[0] == b"doppler":
                self._doppler_hz = value
                self._doppler_at = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            age = None if self._doppler_at is None else time.monotonic() - self._doppler_at
            return {
                "source": self.address,
                "frequency_hz": self._frequency_hz,
                "doppler_hz": self._doppler_hz,
                # O Station Manager anuncia a cada tick (1 s). Parado há mais
                # que alguns segundos = passagem acabou ou satélite abaixo da
                # elevação mínima de apontamento.
                "doppler_age_s": age,
            }
