"""Testes do controle em execução e do painel web.

O painel muda o simulador enquanto ele transmite. O que se confere aqui é que
cada mudança tem o efeito declarado NO SINAL (um emissor desligado some do
espectro, o Doppler começa no instante do pedido), que um pedido inválido não
deixa o simulador meio alterado, e que o HTTP fala de verdade — servidor real
numa porta real, não um handler chamado à mão.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from sdr_sim import emitters as em
from sdr_sim.control import SimController
from sdr_sim.main import build_spectrum as main_build_spectrum
from sdr_sim.main import parse_args
from sdr_sim.packets import PacketMonitor
from sdr_sim.panel import start_panel
from sdr_sim.spectrum import VirtualSpectrum

SAMPLE_RATE = 240_000
CENTER = 145_900_000.0


def carrier_spectrum(offset_hz: float = 20_000.0, snr_db=None) -> VirtualSpectrum:
    return VirtualSpectrum(
        center_frequency_hz=CENTER,
        sample_rate_hz=SAMPLE_RATE,
        emitters=[em.carrier(CENTER + offset_hz, SAMPLE_RATE)],
        snr_db=snr_db,
        seed=1,
    )


def power(samples: np.ndarray) -> float:
    return float(np.mean(np.abs(samples) ** 2))


# --- emissor ligado/desligado -------------------------------------------------


def test_emissor_desligado_nao_e_ouvido():
    spectrum = carrier_spectrum()
    assert power(spectrum.block(4096)) > 0.5

    spectrum.emitters[0].enabled = False
    assert power(spectrum.block(4096)) == 0.0


def test_emissor_desligado_continua_consumindo_a_forma_de_onda():
    """Mesma regra do emissor fora da banda: o satélite não para de
    transmitir só porque ninguém está ouvindo."""
    fs2 = em.fs2_beacon(CENTER, SAMPLE_RATE, 4800, bytes(range(8)))
    spectrum = VirtualSpectrum(CENTER, SAMPLE_RATE, [fs2], snr_db=None)
    fs2.enabled = False

    spectrum.block(1000)

    assert fs2._cursor == 1000


# --- Doppler iniciado no meio da execução ---------------------------------------


def test_passagem_comeca_no_instante_pedido_e_nao_no_zero():
    doppler = em.PassDoppler(3500.0, 100.0, start_s=40.0)

    # No início da passagem o satélite está se aproximando: desvio ~ +máximo.
    assert doppler.shift_at(40.0) == pytest.approx(3500.0, rel=0.01)
    # Na aproximação máxima, zero.
    assert doppler.shift_at(90.0) == pytest.approx(0.0, abs=1e-6)


# --- controlador ---------------------------------------------------------------


def test_apply_muda_sintonia_snr_e_emissor():
    controller = SimController(carrier_spectrum(), 4096)

    controller.apply({
        "tune_hz": CENTER + 5000,
        "snr_db": None,
        "emitters": {"carrier": {"amplitude": 0.5}},
    })

    assert controller.spectrum.center_frequency_hz == CENTER + 5000
    assert controller.spectrum.snr_db is None
    assert controller.emitter("carrier").amplitude == 0.5


def test_pedido_invalido_nao_muda_nada():
    """Tudo-ou-nada: a sintonia válida NÃO pode ser aplicada se outro campo
    do mesmo pedido for inválido."""
    controller = SimController(carrier_spectrum(), 4096)

    with pytest.raises(ValueError, match="amplitude"):
        controller.apply({
            "tune_hz": CENTER + 5000,
            "emitters": {"carrier": {"amplitude": 99}},
        })

    assert controller.spectrum.center_frequency_hz == CENTER


@pytest.mark.parametrize("changes, message", [
    ({"tune_hz": True}, "número"),         # bool é int em Python
    ({"tune_hz": -1}, "positivo"),
    ({"snr_db": "alto"}, "número"),
    ({"snr_db": 500}, "fora"),
    ({"emitters": {"nao-existe": {"enabled": True}}}, "desconhecido"),
    ({"emitters": {"carrier": {"enabled": 1}}}, "true/false"),
    ({"frequencia": 1}, "desconhecido"),
    ([1, 2], "objeto"),
])
def test_pedidos_invalidos_sao_recusados(changes, message):
    controller = SimController(carrier_spectrum(), 4096)

    with pytest.raises(ValueError, match=message):
        controller.apply(changes)


def test_doppler_pelo_painel_comeca_agora():
    fs2 = em.fs2_beacon(CENTER, SAMPLE_RATE, 4800, bytes(range(8)))
    controller = SimController(VirtualSpectrum(CENTER, SAMPLE_RATE, [fs2], snr_db=None), 8192)
    for _ in range(30):
        controller.next_block()
    now = controller.spectrum.elapsed_s

    controller.apply({"doppler": {"max_shift_hz": 3500, "pass_duration_s": 60}})

    assert fs2.doppler.start_s == pytest.approx(now)
    assert fs2.doppler.shift_at(now) == pytest.approx(3500.0, rel=0.01)

    controller.apply({"doppler": {"max_shift_hz": 0}})
    assert fs2.doppler is None


def test_espectro_do_painel_poe_o_pico_no_offset_do_emissor():
    controller = SimController(carrier_spectrum(offset_hz=30_000.0), 8192)
    controller.next_block()

    db = controller.spectrum_db(bins=1024)
    peak_bin = int(np.argmax(db))
    peak_hz = (peak_bin - 512) * SAMPLE_RATE / 1024

    assert peak_hz == pytest.approx(30_000.0, abs=SAMPLE_RATE / 1024)


def test_periodo_do_fs2_e_quadro_mais_silencio():
    """A taxa esperada de pacotes sai daqui: (32+4+64) bytes a 4800 baud mais
    0,5 s de silêncio = 800/4800 + 0,5 s."""
    fs2 = em.fs2_beacon(CENTER, SAMPLE_RATE, 4800, bytes(range(64)), gap_s=0.5)
    controller = SimController(VirtualSpectrum(CENTER, SAMPLE_RATE, [fs2]), 8192)

    assert controller.fs2_period_s() == pytest.approx(800 / 4800 + 0.5, abs=1e-3)


# --- partida: --emitters decide o que começa ligado ------------------------------


def test_os_tres_emissores_existem_e_so_os_pedidos_comecam_ligados():
    spectrum = main_build_spectrum(parse_args(["--emitters", "fs2,carrier"]))

    state = {e.name: e.enabled for e in spectrum.emitters}
    assert state == {"fs2": True, "fm": False, "carrier": True}


# --- monitor de pacotes --------------------------------------------------------


def packet(seq: int, payload: bytes) -> list[bytes]:
    header = json.dumps({"seq": seq, "bit_offset": 100 * seq}).encode()
    return [b"raw_packet", header, payload]


def test_monitor_separa_pacote_certo_de_divergente():
    expected = bytes(range(16))
    monitor = PacketMonitor("tcp://ninguem:1", expected, threading.Event())

    monitor.record(packet(1, expected + b"\xff" * 10))
    monitor.record(packet(2, b"\x00\x01\x02\x07" + b"\x00" * 20))
    monitor.record([b"lixo"])  # envelope errado: ignorado, não conta

    snap = monitor.snapshot()
    assert (snap["total"], snap["ok"], snap["diverge"]) == (2, 1, 1)
    assert snap["recent"][0]["seq"] == 2  # mais recente primeiro

    monitor.reset()
    assert monitor.snapshot()["total"] == 0


def test_monitor_recebe_por_zmq_de_verdade():
    """Thread iniciada de verdade, socket de verdade. Existe porque o teste
    acima chama record() à mão e deixou passar um monitor que quebrava no
    .start() — só a execução no compose pegou."""
    import time

    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    shutdown = threading.Event()
    expected = bytes(range(16))
    monitor = PacketMonitor(f"tcp://127.0.0.1:{port}", expected, shutdown)
    monitor.start()

    try:
        deadline = time.monotonic() + 5
        while monitor.snapshot()["total"] == 0 and time.monotonic() < deadline:
            # Reenvia até chegar: o SUB leva um instante para conectar, e o
            # PUB descarta o que publica antes disso.
            publisher.send_multipart(packet(7, expected))
            time.sleep(0.05)

        assert monitor.snapshot()["ok"] >= 1
    finally:
        shutdown.set()
        monitor.join(timeout=2)
        publisher.close(linger=0)
        context.term()

    assert not monitor.is_alive()


# --- HTTP de verdade -------------------------------------------------------------


@pytest.fixture
def panel():
    controller = SimController(carrier_spectrum(), 4096)
    controller.next_block()
    server = start_panel(controller, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield controller, base
    server.shutdown()
    server.server_close()


def http(method: str, url: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_pagina_e_servida(panel):
    _, base = panel
    status, body = http("GET", base + "/")

    assert status == 200
    assert b"GRS SDR Sim" in body


def test_state_e_spectrum_respondem_json(panel):
    _, base = panel

    status, body = http("GET", base + "/api/state")
    state = json.loads(body)
    assert status == 200
    assert state["center_frequency_hz"] == CENTER
    assert state["emitters"][0]["name"] == "carrier"

    status, body = http("GET", base + "/api/spectrum")
    assert status == 200
    assert len(json.loads(body)) == 1024


def test_post_control_muda_o_simulador(panel):
    controller, base = panel

    status, body = http("POST", base + "/api/control", {"tune_hz": CENTER + 1000})

    assert status == 200
    assert json.loads(body)["center_frequency_hz"] == CENTER + 1000
    assert controller.spectrum.center_frequency_hz == CENTER + 1000


def test_post_invalido_devolve_400_com_o_motivo(panel):
    controller, base = panel

    status, body = http("POST", base + "/api/control", {"snr_db": 999})

    assert status == 400
    assert "snr_db" in json.loads(body)["error"]
    assert controller.spectrum.snr_db is None


def test_rota_desconhecida_devolve_404(panel):
    _, base = panel
    assert http("GET", base + "/nada")[0] == 404
