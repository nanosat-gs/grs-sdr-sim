"""--frame general-telemetry: um quadro NGHam de verdade depois do syncword."""

from __future__ import annotations

import struct

import numpy as np
import pyngham

from sdr_sim import emitters as em
from sdr_sim import fs2_frames
from sdr_sim.main import expected_payload, parse_args

TIMESTAMP = 1_791_201_600


def test_general_telemetry_tem_os_111_bytes_do_firmware():
    packet = fs2_frames.general_telemetry(TIMESTAMP)

    assert len(packet) == 111
    assert packet[:8] == b"\x10 PY0EFS"
    assert struct.unpack_from(">I", packet, 8)[0] == TIMESTAMP
    assert struct.unpack_from(">h", packet, 62)[0] == -250      # corrente da bateria
    assert struct.unpack_from(">hhh", packet, 91) == (-27, -48, 520)
    assert packet[110] == 1                                      # modo


def test_o_quadro_volta_pelo_ngham():
    frame = fs2_frames.ngham_frame(fs2_frames.general_telemetry(TIMESTAMP))

    payload, errors, _ = pyngham.PyNGHam().decode(list(frame))

    assert bytes(payload) == fs2_frames.general_telemetry(TIMESTAMP)
    assert errors == 0
    assert len(frame) == 3 + 159   # size tag + o codeword de 159


def test_modo_general_telemetry_nao_embaralha_duas_vezes():
    args = parse_args(["--frame", "general-telemetry", "--frame-timestamp", str(TIMESTAMP)])

    assert expected_payload(args) == fs2_frames.ngham_frame(fs2_frames.general_telemetry(TIMESTAMP))


def test_padrao_continua_o_de_antes():
    assert expected_payload(parse_args([])) == em.ccsds_scramble(bytes(range(64)))


def test_sem_frame_timestamp_o_pacote_muda_a_cada_rajada():
    from sdr_sim.main import live_telemetry

    assert live_telemetry(parse_args(["--frame", "general-telemetry"])) is True
    assert live_telemetry(parse_args(["--frame", "general-telemetry",
                                      "--frame-timestamp", str(TIMESTAMP)])) is False
    assert live_telemetry(parse_args([])) is False


def decode(frame: bytes) -> bytes:
    payload, errors, _ = pyngham.PyNGHam().decode(list(frame))
    assert errors == 0
    return bytes(payload)


def test_stream_um_pacote_novo_por_chamada():
    now = [TIMESTAMP]
    stream = fs2_frames.GeneralTelemetryStream(clock=lambda: now[0])

    first = decode(stream())
    now[0] += 7
    second = decode(stream())

    assert first != second
    assert struct.unpack_from(">I", first, 8)[0] == TIMESTAMP
    assert struct.unpack_from(">I", second, 8)[0] == TIMESTAMP + 7
    # O contador visível: página do log de dados, 1000 + sequência.
    assert struct.unpack_from(">I", first, 75)[0] == 1000
    assert struct.unpack_from(">I", second, 75)[0] == 1001


def test_valores_variam_e_a_corrente_da_bateria_troca_de_sinal():
    """Num quarto de período a bateria carrega (corrente positiva); em três
    quartos, descarrega. É o que faz o int16 do decoder aparecer na tela."""
    period = fs2_frames.VARIATION_PERIOD_S

    def battery_current(t):
        return struct.unpack_from(">h", fs2_frames.general_telemetry(t), 62)[0]

    def obdh_temp(t):
        return struct.unpack_from(">H", fs2_frames.general_telemetry(t), 12)[0]

    assert battery_current(TIMESTAMP + period // 4) == 150
    assert battery_current(TIMESTAMP + 3 * period // 4) == -650
    assert obdh_temp(TIMESTAMP + period // 4) != obdh_temp(TIMESTAMP + 3 * period // 4)


def test_emissor_regenera_a_forma_de_onda_a_cada_rajada():
    calls = []

    def source():
        calls.append(len(calls))
        return fs2_frames.ngham_frame(fs2_frames.general_telemetry(TIMESTAMP + len(calls)))

    emitter = em.fs2_beacon(468.4e6, 240_000, 4800, source(), gap_s=0.1, payload_source=source)
    period = len(emitter.waveform)
    emitter.take(3 * period)

    assert emitter.bursts_sent == 3
    assert len(calls) == 1 + 3   # a forma de onda inicial + uma por rajada


def test_conferencia_do_monitor_quando_o_pacote_muda():
    stream = fs2_frames.GeneralTelemetryStream(clock=lambda: TIMESTAMP)

    assert fs2_frames.is_general_telemetry_frame(stream() + bytes(96))
    assert not fs2_frames.is_general_telemetry_frame(em.ccsds_scramble(bytes(range(64))))


def test_monitor_aceita_funcao_de_conferencia():
    import threading

    from sdr_sim.packets import PacketMonitor

    monitor = PacketMonitor("tcp://ninguem:1", fs2_frames.is_general_telemetry_frame, threading.Event())
    good = fs2_frames.GeneralTelemetryStream(clock=lambda: TIMESTAMP)()
    monitor.record([b"raw_packet", b"{}", good + bytes(96)])
    monitor.record([b"raw_packet", b"{}", bytes(258)])

    snapshot = monitor.snapshot()
    assert (snapshot["ok"], snapshot["diverge"]) == (1, 1)


def test_quadro_nao_contem_o_syncword():
    """Um syncword (a até 1 bit, a tolerância do detector) dentro do quadro
    faria o detector achar um pacote fantasma no meio dele."""
    bits = em.bytes_to_bits(fs2_frames.ngham_frame(fs2_frames.general_telemetry(TIMESTAMP)))
    sync = em.bytes_to_bits(em.NGHAM_SYNCWORD)
    windows = np.lib.stride_tricks.sliding_window_view(bits, len(sync))

    assert np.min(np.count_nonzero(windows != sync, axis=1)) > 1
