"""O rádio de bordo (TTC_1) ouvindo o uplink: IQ GMSK -> bits -> syncword -> NGHam.

Escrito da definição, e de propósito sem nada do `grs-uplink` (que modula) nem
do `grs-demodulator` (que demodula o downlink na estação): se este receptor
recuperar o pacote, é porque a forma de onda da estação é GMSK de verdade — e
não porque os dois lados compartilham a mesma convenção acidental.

O receptor:

1. discriminador de frequência: a fase entre amostras vizinhas;
2. tira o desvio médio da rajada (Doppler residual + erro de oscilador). O
   preâmbulo alternado e o codeword embaralhado deixam os bits equilibrados,
   então a média do desvio é o desvio da portadora;
3. integra e decide um símbolo por período, testando várias fases de
   amostragem (a rajada não chega alinhada ao relógio de bit);
4. procura o syncword do NGHam (5D E6 2A 7E, MSB primeiro) bit a bit,
   tolerando poucos bits errados;
5. entrega os bytes depois do syncword ao PyNGHam.

`rx_max_bytes` emula o TTC2 lendo um número fixo de bytes por pacote
(`radio_recv(rx_packet, 80U, ...)` no `uplink_manager.c` v1.0 — o TODO-1 do
planejamento). 0 = sem limite, a premissa atual do plano.
"""

from __future__ import annotations

import numpy as np
import pyngham

NGHAM_SYNCWORD = bytes((0x5D, 0xE6, 0x2A, 0x7E))
SYNC_BITS = np.unpackbits(np.frombuffer(NGHAM_SYNCWORD, dtype=np.uint8))
# Size tag (3) + o maior codeword (255).
MAX_FRAME_BYTES = 258
# Fases de amostragem testadas por símbolo.
PHASES = 16


def _symbol_sums(frequency: np.ndarray, samples_per_symbol: int, offset: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(frequency)))
    starts = np.arange(offset, len(frequency) - samples_per_symbol + 1, samples_per_symbol)
    return cumulative[starts + samples_per_symbol] - cumulative[starts]


def candidate_bitstreams(iq: np.ndarray, samples_per_symbol: int) -> list[np.ndarray]:
    """Uma sequência de bits por fase de amostragem, a do meio primeiro."""
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2 * samples_per_symbol:
        return []
    frequency = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float64)
    frequency -= frequency.mean()
    step = max(1, samples_per_symbol // PHASES)
    offsets = sorted(range(0, samples_per_symbol, step),
                     key=lambda o: abs(o - samples_per_symbol // 2))
    return [(_symbol_sums(frequency, samples_per_symbol, o) > 0).astype(np.uint8) for o in offsets]


def find_syncwords(bits: np.ndarray, max_errors: int = 2) -> list[int]:
    """Índices do primeiro bit DEPOIS de cada syncword achado."""
    if len(bits) < len(SYNC_BITS):
        return []
    windows = np.lib.stride_tricks.sliding_window_view(bits, len(SYNC_BITS))
    errors = (windows != SYNC_BITS).sum(axis=1)
    found, last = [], -len(SYNC_BITS)
    for index in np.flatnonzero(errors <= max_errors):
        if index - last >= len(SYNC_BITS):    # o mesmo syncword em índices vizinhos
            found.append(int(index) + len(SYNC_BITS))
            last = index
    return found


def ngham_decode(after_sync: bytes) -> bytes | None:
    payload, _, _ = pyngham.PyNGHam().decode(list(after_sync))
    return bytes(payload) if payload else None


def ngham_decode_frame(after_sync: bytes) -> tuple[bytes, bytes] | None:
    """(pacote, quadro cru): o quadro é o size tag + codeword, só os bytes que
    o NGHam consumiu, exatamente como chegaram — com os erros que o
    Reed-Solomon corrigiu. Comparado ao quadro recodificado, diz quantos bits
    a recepção errou (o modo bancada registra isso)."""
    decoder = pyngham.PyNGHam()
    for index, byte in enumerate(after_sync):
        payload, _, _ = decoder.decode_byte(byte)
        if payload:
            return bytes(payload), bytes(after_sync[:index + 1])
    return None


def receive_frames(iq: np.ndarray, sample_rate_hz: int, baud: int,
                   rx_max_bytes: int = 0) -> list[tuple[bytes, bytes]]:
    """(pacote, quadro cru) de cada quadro que o rádio de bordo tira de uma rajada."""
    if sample_rate_hz % baud:
        raise ValueError(f"{sample_rate_hz} S/s não é múltiplo de {baud} baud")
    for bits in candidate_bitstreams(iq, sample_rate_hz // baud):
        frames = []
        for start in find_syncwords(bits):
            after = np.packbits(bits[start:start + 8 * MAX_FRAME_BYTES]).tobytes()
            if rx_max_bytes:
                # O rádio leu só isto; o resto do buffer de bordo fica zerado.
                after = after[:rx_max_bytes].ljust(MAX_FRAME_BYTES, b"\x00")
            decoded = ngham_decode_frame(after)
            if decoded is not None:
                frames.append(decoded)
        if frames:
            return frames
    return []


def receive(iq: np.ndarray, sample_rate_hz: int, baud: int, rx_max_bytes: int = 0) -> list[bytes]:
    """Os pacotes que o rádio de bordo tira de uma rajada (normalmente um)."""
    return [packet for packet, _ in receive_frames(iq, sample_rate_hz, baud, rx_max_bytes)]
