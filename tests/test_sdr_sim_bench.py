"""Modo bancada: o simulador como modem do rádio do Pi, o rádio falso e o gateway.

O rádio de verdade (Si4463 + Bluepill) não entra aqui: no lugar dele, um
rádio de mentira com a mesma API (`FakeRadio`) ou o `bench_radio_stub`, que
responde com o `satellite.py`. O que se confere é o contrato: o quadro que
chega ao rádio é o que a estação transmitiu, o `ack` só sai com a
confirmação do rádio e com os horários dele, e o que o rádio recebe entra na
cadeia do canal certo.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import threading
import time

import numpy as np
import pyngham
import pytest
import zmq

from sdr_sim import bench, fs2_frames, gateway, main, modulation, uplink_rx
from sdr_sim import emitters as em
from sdr_sim.bench_radio_stub import EPS_BEACON_ID, StubRadio
from sdr_sim.satellite import SatelliteModel

RATE = 240_000
BAUD = 1200
SPS = RATE // BAUD
NOMINAL = 145_900_000.0
KEYS = {"activate_module": b"TESTE-actv-mod01", "deactivate_module": b"TESTE-deac-mod01"}
STATION = b" PY0SIM"
NOW = 1_791_201_600


def ping() -> bytes:
    return b"\x40" + STATION


def module_tc(tc_id: int = 0x45, module: int = 3, key: bytes = KEYS["activate_module"]) -> bytes:
    body = bytes([tc_id]) + STATION + bytes([module])
    return body + hmac.new(key, body, hashlib.sha1).digest()


def encoded(packet: bytes) -> bytes:
    """O quadro inteiro do PyNGHam (preâmbulo + sync + size tag + codeword)."""
    return bytes(pyngham.PyNGHam().encode(list(packet)))


def uplink_iq(packet: bytes) -> np.ndarray:
    iq = modulation.modulate_2gfsk(em.bytes_to_bits(encoded(packet)), SPS, bt=0.5)
    return iq.astype(np.complex64)


def burst(packet: bytes, burst_id: str = "7"):
    iq = uplink_iq(packet)
    header = {"burst_id": burst_id, "sample_rate": RATE, "tx_freq_hz": NOMINAL,
              "n_samples": len(iq)}
    return header, iq.tobytes()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeRadio:
    def __init__(self, reply: dict | None = None) -> None:
        self.calls: list[tuple[str, bytes, float]] = []
        self.reply = reply or {"ok": True, "started_at": "2026-11-24T12:00:00.100000+00:00",
                               "ended_at": "2026-11-24T12:00:00.500000+00:00"}

    def tx(self, channel: str, frame: bytes, timeout_s: float) -> dict:
        self.calls.append((channel, frame, timeout_s))
        return self.reply


def station(radio, monitor=None) -> bench.BenchUplinkStation:
    return bench.BenchUplinkStation(None, NOMINAL, BAUD, radio, threading.Event(), monitor=monitor)


# --- o receptor de uplink devolve o quadro cru -------------------------------------


def test_receive_frames_devolve_o_quadro_cru_do_tamanho_certo():
    packet = module_tc()
    [(got_packet, raw)] = uplink_rx.receive_frames(uplink_iq(packet), RATE, BAUD)
    assert got_packet == packet
    clean = fs2_frames.ngham_frame(packet)
    # Tudo depois do syncword, nem um byte a mais.
    assert len(raw) == len(clean)
    # A rajada acaba no último símbolo e o filtro gaussiano deixa o último bit
    # sem cauda: o cru pode chegar com esse bit errado, que o RS corrige.
    # É por isso que o modo bancada manda o quadro recodificado.
    assert bench.corrected_bits(raw, clean) <= 1
    assert uplink_rx.receive(uplink_iq(packet), RATE, BAUD) == [packet]


def test_bits_corrigidos():
    assert bench.corrected_bits(bytes([0x00, 0xFF]), bytes([0x01, 0xFF])) == 1
    assert bench.corrected_bits(bytes(1), bytes(2)) == 16


def test_tempo_no_ar_conta_preambulo_e_sync_do_chip():
    # 50 B a 1200 baud: (4 + 4 + 50) * 8 / 1200
    assert bench.airtime_s("vhf", 50) == pytest.approx(58 * 8 / 1200)
    assert bench.airtime_s("uhf", 258) == pytest.approx(266 * 8 / 4800)


# --- a :5559 no modo bancada -------------------------------------------------------


def test_ack_so_com_a_confirmação_do_rádio_e_com_os_horários_dele():
    radio = FakeRadio()
    reply, after = station(radio).handle_burst(*burst(ping()))

    assert after is None
    assert reply["status"] == "ack"
    assert reply["burst_id"] == "7"
    assert reply["started_at"] == radio.reply["started_at"]
    assert reply["ended_at"] == radio.reply["ended_at"]
    assert reply["actual_freq_hz"] == NOMINAL
    [(channel, frame, timeout)] = radio.calls
    assert channel == "vhf"
    # Vai ao rádio o quadro recodificado, sem os erros da recepção.
    assert frame == fs2_frames.ngham_frame(ping())
    assert "corrigiu" in reply["detail"]
    assert timeout == pytest.approx(bench.airtime_s("vhf", len(frame)) + bench.TX_MARGIN_S)


def test_erro_do_rádio_vira_error_com_o_motivo():
    radio = FakeRadio({"ok": False, "error": "FIFO underflow"})
    reply, _ = station(radio).handle_burst(*burst(ping()))
    assert reply["status"] == "error"
    assert "FIFO underflow" in reply["detail"]


def test_rádio_sem_resposta_vira_timeout():
    radio = FakeRadio({"ok": False, "timeout": True, "error": "o rádio não respondeu"})
    reply, _ = station(radio).handle_burst(*burst(ping()))
    assert reply["status"] == "timeout"


def test_rajada_que_não_decodifica_não_vai_ao_rádio():
    radio = FakeRadio()
    noise = (np.random.default_rng(1).normal(size=RATE) * 0.5).astype(np.complex64)
    header = {"burst_id": "9", "sample_rate": RATE, "tx_freq_hz": NOMINAL}
    reply, _ = station(radio).handle_burst(header, noise.tobytes())
    assert reply["status"] == "error"
    assert "não decodificou" in reply["detail"]
    assert radio.calls == []


def test_rádio_fora_do_ar_recusa_na_hora_sem_tentar():
    class DeadClient:
        address = "tcp://pi:5600"

        def health(self):
            return None

    monitor = bench.RadioMonitor(DeadClient(), threading.Event())
    monitor.poll_once()
    radio = FakeRadio()
    reply, _ = station(radio, monitor).handle_burst(*burst(ping()))
    assert reply["status"] == "error"
    assert "fora do ar" in reply["detail"]
    assert radio.calls == []


def test_tx_desligado_continua_valendo_no_modo_bancada():
    radio = FakeRadio()
    uplink = station(radio)
    uplink.tx_enabled = False
    reply, _ = uplink.handle_burst(*burst(ping()))
    assert reply["status"] == "tx_disabled"
    assert radio.calls == []


def test_injeção_de_falha_é_da_bluepill_no_modo_bancada():
    with pytest.raises(ValueError, match="Bluepill"):
        station(FakeRadio()).apply({"fail_next": -2})


def test_snapshot_do_modo_bancada_tem_os_horários_do_rádio():
    radio = FakeRadio()
    uplink = station(radio)
    uplink.handle_burst(*burst(ping()))
    snap = uplink.snapshot()
    assert snap["mode"] == "bancada"
    assert snap["frames_sent"] == 1
    [recent] = snap["recent_bursts"]
    assert recent["radio_started_at"] == radio.reply["started_at"]
    assert recent["packets"] == ["0x40 (8 B)"]


# --- os quadros que o rádio recebe -------------------------------------------------


def test_quadro_do_canal_entra_na_fila_e_o_do_outro_canal_não():
    queued: list[bytes] = []
    receiver = bench.BenchReceiver("tcp://x:1", "uhf", queued.append, threading.Event())
    frame = fs2_frames.ngham_frame(fs2_frames.general_telemetry(NOW))
    meta = json.dumps({"rssi": 187, "received_at": "2026-11-24T12:00:01+00:00"}).encode()

    assert not receiver.handle([b"vhf", frame, meta])
    assert receiver.handle([b"uhf", frame, meta])

    assert queued == [frame]
    [recent] = receiver.snapshot()["recent"]
    assert (recent["pkt_id"], recent["rssi"], recent["bytes"]) == ("0x10", 187, len(frame))


def test_quadro_do_rádio_vira_iq_que_o_receptor_decodifica():
    """O emissor fs2 calado, um quadro na fila: a rajada sai, e é o mesmo pacote."""
    emitter = em.fs2_beacon(NOMINAL, RATE, BAUD, b"\x00" * 8)
    emitter.continuous = False
    receiver = bench.BenchReceiver("tcp://x:1", "vhf", emitter.enqueue, threading.Event())
    pong = bytes([0x02]) + fs2_frames.CALLSIGN + STATION

    assert not np.any(emitter.take(RATE // 10))          # calado antes do quadro
    receiver.handle([b"vhf", fs2_frames.ngham_frame(pong)])
    iq = emitter.take(3 * RATE)
    assert uplink_rx.receive(iq, RATE, BAUD) == [pong]


def test_receptor_assina_o_pub_do_rádio():
    port = free_port()
    context = zmq.Context.instance()
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    shutdown = threading.Event()
    got = threading.Event()
    receiver = bench.BenchReceiver(f"tcp://127.0.0.1:{port}", "vhf",
                                   lambda frame: got.set(), shutdown)
    receiver.start()
    try:
        frame = fs2_frames.ngham_frame(ping())
        deadline = time.time() + 5
        while not got.is_set() and time.time() < deadline:
            pub.send_multipart([b"vhf", frame, b"{}"])
            got.wait(0.2)
        assert got.is_set()
    finally:
        shutdown.set()
        receiver.join(timeout=2)
        pub.close(linger=0)


# --- o rádio falso ------------------------------------------------------------------


class Published:
    def __init__(self) -> None:
        self.frames: list[tuple[str, bytes, dict]] = []

    def __call__(self, channel: str, frame: bytes, meta: dict) -> None:
        self.frames.append((channel, frame, meta))

    def packets(self) -> list[tuple[str, bytes]]:
        return [(c, uplink_rx.ngham_decode(f)) for c, f, _ in self.frames]


def stub() -> tuple[StubRadio, Published]:
    published = Published()
    radio = StubRadio(SatelliteModel(KEYS, clock=lambda: NOW), published,
                      response_delay_s=0.0, sleep=lambda s: None)
    return radio, published


def stub_tx(radio: StubRadio, packet: bytes) -> dict:
    reply = radio.handle({"op": "tx", "channel": "vhf",
                          "frame_hex": fs2_frames.ngham_frame(packet).hex()})
    # O satélite responde numa thread: espera ela terminar.
    if radio.responder is not None:
        radio.responder.join(timeout=2)
    return reply


def test_stub_ping_responde_0x02_pela_vhf():
    radio, published = stub()
    reply = stub_tx(radio, ping())
    assert reply["ok"] and reply["started_at"] <= reply["ended_at"]
    [(channel, packet)] = published.packets()
    assert channel == "vhf"
    assert packet == bytes([0x02]) + fs2_frames.CALLSIGN + STATION


def test_stub_activate_responde_0x13_pela_uhf_e_chave_errada_cala():
    radio, published = stub()
    stub_tx(radio, module_tc())
    stub_tx(radio, module_tc(key=b"chave-errada-000"))
    [(channel, packet)] = published.packets()
    assert channel == "uhf"
    assert packet[0] == 0x13


def test_stub_fail_next_devolve_o_código_no_0x13():
    radio, published = stub()
    assert radio.handle({"op": "sat", "fail_next": -2})["ok"]
    stub_tx(radio, module_tc())
    [(_, packet)] = published.packets()
    assert struct.unpack(">h", packet[-2:])[0] == -2


def test_stub_beacon_na_vhf_e_0x10_logo_depois_na_uhf():
    radio, published = stub()
    radio.send_beacon()
    [(c1, beacon), (c2, telemetry)] = published.packets()
    assert (c1, beacon[0]) == ("vhf", EPS_BEACON_ID)
    assert (c2, telemetry[0]) == ("uhf", fs2_frames.GENERAL_TELEMETRY_ID)


def test_stub_recusa_pedido_inválido():
    radio, _ = stub()
    assert not radio.handle({"op": "tx", "channel": "shf", "frame_hex": "00"})["ok"]
    assert not radio.handle({"op": "tx", "channel": "vhf", "frame_hex": "zz"})["ok"]
    assert not radio.handle({"op": "nada"})["ok"]
    assert radio.handle({"op": "health"})["stub"] is True


def test_cliente_fala_com_o_stub_pela_rede():
    """RadioClient -> REP do stub: o mesmo caminho que a :5559 usa no Pi."""
    port = free_port()
    context = zmq.Context.instance()
    rep = context.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    radio, _ = stub()

    def serve(n: int) -> None:
        for _ in range(n):
            rep.send(json.dumps(radio.handle(json.loads(rep.recv()))).encode())

    server = threading.Thread(target=serve, args=(2,), daemon=True)
    server.start()
    client = bench.RadioClient(f"tcp://127.0.0.1:{port}")
    try:
        assert client.health()["ok"]
        assert client.tx("vhf", fs2_frames.ngham_frame(ping()), 5.0)["ok"]
    finally:
        server.join(timeout=2)
        rep.close(linger=0)


def test_cliente_sem_rádio_devolve_timeout_em_vez_de_travar():
    client = bench.RadioClient(f"tcp://127.0.0.1:{free_port()}")
    started = time.monotonic()
    result = client.tx("vhf", b"\x00" * 10, 0.3)
    assert result["timeout"] and not result["ok"]
    assert time.monotonic() - started < 2.0
    assert client.health(0.2) is None


# --- o gateway ----------------------------------------------------------------------


def test_gateway_sem_iq_recusa_na_hora_com_o_motivo():
    relay = gateway.IqRelay("tcp://pi:5556", "tcp://*:0", threading.Event(), zmq.Context.instance())
    tx = gateway.TxRelay("tcp://pi:5559", relay, zmq.Context.instance())
    reply = json.loads(tx.handle([json.dumps({"burst_id": 3}).encode(), b""]))
    assert reply["status"] == "error"
    assert reply["burst_id"] == "3"          # texto, como o simulador e o USRP
    assert "sem IQ" in reply["detail"]


def test_gateway_iq_parado_há_tempo_também_recusa():
    relay = gateway.IqRelay("tcp://pi:5556", "tcp://*:0", threading.Event(), zmq.Context.instance())
    relay.last_at = time.monotonic() - 10
    tx = gateway.TxRelay("tcp://pi:5559", relay, zmq.Context.instance(), stale_after_s=3)
    reply = json.loads(tx.handle([json.dumps({"burst_id": 4}).encode(), b""]))
    assert reply["status"] == "error" and "há 10 s" in reply["detail"]


def test_gateway_repassa_o_iq_e_o_tx_ao_pi():
    context = zmq.Context.instance()
    pi_iq, pi_tx, chain = free_port(), free_port(), free_port()
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{pi_iq}")
    rep = context.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{pi_tx}")
    shutdown = threading.Event()
    relay = gateway.IqRelay(f"tcp://127.0.0.1:{pi_iq}", f"tcp://127.0.0.1:{chain}", shutdown, context)
    relay.start()
    consumer = context.socket(zmq.SUB)
    consumer.setsockopt(zmq.SUBSCRIBE, b"")
    consumer.connect(f"tcp://127.0.0.1:{chain}")
    try:
        block = np.arange(16, dtype=np.complex64).tobytes()
        deadline = time.time() + 5
        received = None
        while received is None and time.time() < deadline:
            pub.send(block)
            if consumer.poll(200):
                received = consumer.recv()
        assert received == block          # o IQ passa intacto

        answer = {"burst_id": 5, "status": "ack", "started_at": "x", "ended_at": "y"}

        def pi() -> None:
            parts = rep.recv_multipart()
            assert json.loads(parts[0])["burst_id"] == 5 and parts[1] == b"iq"
            rep.send(json.dumps(answer).encode())

        server = threading.Thread(target=pi, daemon=True)
        server.start()
        tx = gateway.TxRelay(f"tcp://127.0.0.1:{pi_tx}", relay, context)
        header = {"burst_id": 5, "sample_rate": RATE, "n_samples": 100}
        assert json.loads(tx.handle([json.dumps(header).encode(), b"iq"])) == answer
        server.join(timeout=2)
    finally:
        shutdown.set()
        relay.join(timeout=2)
        for s in (pub, rep, consumer):
            s.close(linger=0)


def test_gateway_pi_que_não_responde_vira_timeout():
    context = zmq.Context.instance()
    relay = gateway.IqRelay("tcp://pi:5556", "tcp://*:0", threading.Event(), context)
    relay.last_at = time.monotonic()
    tx = gateway.TxRelay(f"tcp://127.0.0.1:{free_port()}", relay, context, margin_s=0.3)
    header = {"burst_id": 6, "sample_rate": RATE, "n_samples": 0}
    reply = json.loads(tx.handle([json.dumps(header).encode(), b""]))
    assert reply["status"] == "timeout"


# --- linha de comando ---------------------------------------------------------------


def test_modo_bancada_recusa_opções_do_satélite_simulado():
    with pytest.raises(SystemExit):
        main.parse_args(["--bench-rx", "tcp://pi:5601", "--sat-keys-file", "k.json"])
    with pytest.raises(SystemExit):
        main.parse_args(["--tx-bind", "tcp://*:5559", "--bench-rx", "tcp://pi:5601"])
    args = main.parse_args(["--tx-bind", "tcp://*:5559", "--bench-radio", "tcp://pi:5600",
                            "--bench-rx", "tcp://pi:5601", "--responses-channel", "vhf"])
    assert args.bench and args.responses_channel == "vhf"


def test_modo_bancada_modula_com_o_preambulo_do_si4463():
    """A estação tem de ver os 4 bytes de preâmbulo que a Bluepill transmite,
    não os 32 de folga do simulador: com 32, o M&M cego no preâmbulo passava
    na bancada e perdia quadros do satélite de verdade."""
    bancada = main.parse_args(["--bench-rx", "tcp://pi:5601", "--responses-channel", "uhf"])
    assert bancada.preamble_bytes == bench.PREAMBLE_BYTES == 4
    assert main.parse_args([]).preamble_bytes == 32
    explicito = main.parse_args(["--bench-rx", "tcp://pi:5601", "--preamble-bytes", "8"])
    assert explicito.preamble_bytes == 8
    with pytest.raises(SystemExit):
        main.parse_args(["--preamble-bytes", "-1"])


def test_gateway_sem_host_do_pi_explica_o_que_falta(capsys):
    with pytest.raises(SystemExit):
        gateway.main(["--iq-source", "tcp://:5556"])
    assert "BANCADA_PI_HOST" in capsys.readouterr().err


def test_gateway_republica_o_tune_para_o_pi_assinar():
    context = zmq.Context.instance()
    synth, out = free_port(), free_port()
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{synth}")
    shutdown = threading.Event()
    relay = gateway.TuneRelay(f"tcp://127.0.0.1:{synth}", f"tcp://127.0.0.1:{out}", shutdown, context)
    relay.start()
    pi = context.socket(zmq.SUB)
    pi.setsockopt(zmq.SUBSCRIBE, b"tune")
    pi.connect(f"tcp://127.0.0.1:{out}")
    try:
        got = None
        deadline = time.time() + 5
        while got is None and time.time() < deadline:
            pub.send_multipart([b"tune", b"145903210"])
            if pi.poll(200):
                got = pi.recv_multipart()
        # O mesmo envelope que o TuneListener do simulador espera.
        assert got == [b"tune", b"145903210"]
    finally:
        shutdown.set()
        relay.join(timeout=2)
        pub.close(linger=0)
        pi.close(linger=0)
