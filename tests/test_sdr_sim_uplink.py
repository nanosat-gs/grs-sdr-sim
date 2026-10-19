"""O satélite simulado recebendo telecomandos: rádio de bordo, OBDH e o caminho das respostas.

O IQ de uplink aqui é montado com a modulação do PRÓPRIO simulador (mesma
definição de GMSK). A prova cruzada — a rajada do `grs-uplink` sendo
recebida por este receptor — fica no ponta a ponta do grs-station
(tests/test_uplink_loop_offline_e2e.py), que enxerga os dois repositórios.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import threading

import numpy as np
import pyngham
import pytest
import zmq

from sdr_sim import emitters as em
from sdr_sim import fs2_frames, modulation, uplink_rx
from sdr_sim.control import SimController
from sdr_sim.satellite import SatelliteModel
from sdr_sim.spectrum import VirtualSpectrum
from sdr_sim.uplink import TxServer, UplinkStation, is_response_frame

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


def uplink_iq(packet: bytes, offset_hz: float = 0.0) -> np.ndarray:
    """O que a estação transmite: o quadro NGHam inteiro do PyNGHam em GMSK."""
    frame = bytes(pyngham.PyNGHam().encode(list(packet)))
    iq = modulation.modulate_2gfsk(em.bytes_to_bits(frame), SPS, bt=0.5)
    if offset_hz:
        iq = iq * np.exp(2j * np.pi * offset_hz * np.arange(len(iq)) / RATE)
    return iq.astype(np.complex64)


def satellite() -> SatelliteModel:
    return SatelliteModel(KEYS, clock=lambda: NOW)


# --- OBDH (process_tc.c v1.0) -------------------------------------------------------


def test_ping_responde_pong_pela_vhf_com_quem_pediu():
    [response] = satellite().process(ping())

    assert response.radio == "vhf"
    assert response.packet == b"\x02 PY0EFS" + STATION


@pytest.mark.parametrize("tc_id", [0x45, 0x46])
def test_modulo_autenticado_responde_feedback_pela_uhf(tc_id):
    key = KEYS["activate_module" if tc_id == 0x45 else "deactivate_module"]
    model = satellite()

    [response] = model.process(module_tc(tc_id, key=key))

    assert response.radio == "uhf"
    assert response.packet == b"\x13 PY0EFS" + STATION + struct.pack(">BIh", tc_id, NOW, 0)
    assert len(response.packet) == 22
    assert model.last_valid_tc == tc_id


def test_hmac_invalido_e_silencio():
    model = satellite()

    assert model.process(module_tc(key=b"chave-errada-123")) == []
    assert model.rejected == 1 and model.last_valid_tc == 0


def test_chave_de_outro_tc_e_silencio():
    assert satellite().process(module_tc(0x46, key=KEYS["activate_module"])) == []


def test_modulo_desconhecido_e_silencio_mas_conta_como_valido():
    """O `default: break` do process_tc_activate_module."""
    model = satellite()

    assert model.process(module_tc(module=9)) == []
    assert model.last_valid_tc == 0x45


def test_tc_desconhecido_e_curto_sao_silencio():
    model = satellite()

    assert model.process(b"\x4a" + STATION + bytes(20)) == []
    assert model.process(b"\x40 PY") == []
    assert model.process(module_tc()[:28]) == []


def test_injecao_de_falha_vale_para_o_proximo_so():
    model = satellite()
    model.fail_next(-2)

    [failed] = model.process(module_tc())
    [ok] = model.process(module_tc())

    assert struct.unpack(">h", failed.packet[20:22])[0] == -2
    assert struct.unpack(">h", ok.packet[20:22])[0] == 0


def test_ignorar_o_proximo():
    model = satellite()
    model.drop_next()

    assert model.process(ping()) == []
    assert len(model.process(ping())) == 1


# --- rádio de bordo -------------------------------------------------------------------


@pytest.mark.parametrize("offset_hz", [0.0, 700.0, -1200.0])
def test_receptor_de_bordo_recupera_o_pacote(offset_hz):
    iq = uplink_iq(module_tc(), offset_hz)
    noise = np.random.default_rng(3).normal(scale=0.15, size=(len(iq), 2)) @ [1, 1j]

    assert uplink_rx.receive((iq + noise).astype(np.complex64), RATE, BAUD) == [module_tc()]


def test_rajada_sem_syncword_nao_da_nada():
    iq = modulation.modulate_2gfsk(em.bytes_to_bits(bytes(range(60))), SPS)

    assert uplink_rx.receive(iq, RATE, BAUD) == []


def test_limite_de_80_bytes_do_ttc2_emulado():
    """TODO-1: com o rádio lendo 80 bytes, a classe 2 (82 B no ar) passa pelo
    Reed-Solomon e a classe 3 (114 B, o Update TLE) não. Sem limite, as duas."""
    class_2 = module_tc()                                   # 29 B
    class_3 = b"\x4f" + STATION + bytes(50) + bytes(20)     # 78 B, o tamanho do Update TLE

    assert uplink_rx.receive(uplink_iq(class_2), RATE, BAUD, rx_max_bytes=80) == [class_2]
    assert uplink_rx.receive(uplink_iq(class_3), RATE, BAUD, rx_max_bytes=80) == []
    assert uplink_rx.receive(uplink_iq(class_3), RATE, BAUD) == [class_3]


# --- a :5559 e o caminho das respostas ---------------------------------------------


class FixedDoppler:
    def __init__(self, shift_hz: float, audible: bool = True) -> None:
        self.shift_hz = shift_hz
        self.audible = audible

    def shift_at(self, elapsed_s, carrier_hz=None):
        return self.shift_hz

    def audible_at(self, elapsed_s):
        return self.audible


def station(doppler=None, **kwargs):
    fs2 = em.fs2_beacon(NOMINAL, RATE, BAUD, bytes(16), doppler=doppler)
    controller = SimController(VirtualSpectrum(NOMINAL, RATE, [fs2], snr_db=None, seed=1), 4096)
    sent: list[tuple[str, bytes]] = []
    done = threading.Event()
    done.set()      # handle_burst espera a duração da rajada com shutdown.wait: aqui, não espera
    uplink = UplinkStation(SatelliteModel(KEYS, clock=lambda: NOW), controller, NOMINAL, BAUD,
                           respond=lambda radio, frame: sent.append((radio, frame)),
                           shutdown=done, **kwargs)
    controller.uplink = uplink
    return uplink, controller, sent


def burst(packet: bytes, tx_freq: float = NOMINAL, burst_id: str = "7"):
    return {"burst_id": burst_id, "sample_rate": RATE, "tx_freq_hz": tx_freq}, uplink_iq(packet).tobytes()


def test_rajada_recebe_ack_e_o_satelite_responde():
    uplink, _, sent = station()

    reply, deliver = uplink.handle_burst(*burst(module_tc()))
    deliver()

    assert (reply["burst_id"], reply["status"], reply["actual_freq_hz"]) == ("7", "ack", NOMINAL)
    assert reply["started_at"] and reply["ended_at"]
    [(radio, frame)] = sent
    assert radio == "uhf"
    assert bytes(pyngham.PyNGHam().decode(list(frame))[0])[:1] == b"\x13"
    assert is_response_frame(frame)


def test_doppler_corrigido_e_ouvido_sem_correcao_nao():
    """O satélite vindo na direção da estação: +3 kHz. Transmitir na nominal
    chega 3 kHz acima e o rádio de bordo não ouve; em nominal - 3 kHz, ouve."""
    uplink, _, sent = station(doppler=FixedDoppler(3000.0))

    uplink.handle_burst(*burst(ping(), NOMINAL))[1]()
    assert sent == []
    assert "fora da banda" in uplink.snapshot()["recent_bursts"][0]["outcome"]

    uplink.handle_burst(*burst(ping(), NOMINAL - 3000.0))[1]()
    assert [radio for radio, _ in sent] == ["vhf"]


def test_abaixo_do_horizonte_nao_ouve():
    uplink, _, sent = station(doppler=FixedDoppler(0.0, audible=False))

    reply, deliver = uplink.handle_burst(*burst(ping()))
    deliver()

    assert reply["status"] == "ack"       # a estação transmitiu; quem não ouviu foi o satélite
    assert sent == []


@pytest.mark.parametrize("header, iq_bytes, detail", [
    ({"burst_id": "1"}, 8, "sample_rate"),
    ({"burst_id": "1", "sample_rate": RATE, "tx_freq_hz": NOMINAL}, 7, "cf32_le"),
    ({"burst_id": "1", "sample_rate": 1000, "tx_freq_hz": NOMINAL}, 8 * 20_000, "limite"),
], ids=["sem-taxa", "iq-torto", "longa-demais"])
def test_pedido_invalido_e_erro(header, iq_bytes, detail):
    reply, deliver = station()[0].handle_burst(header, bytes(iq_bytes))

    assert reply["status"] == "error" and detail in reply["detail"] and deliver is None


def test_tx_desligado_pelo_painel():
    uplink, controller, _ = station()
    controller.apply({"uplink": {"tx_enabled": False}})

    assert uplink.handle_burst(*burst(ping()))[0]["status"] == "tx_disabled"


def test_painel_injeta_falha_e_mostra_o_estado():
    uplink, controller, sent = station()
    controller.apply({"uplink": {"fail_next": -2}})

    uplink.handle_burst(*burst(module_tc()))[1]()

    state = controller.state()["uplink"]
    assert state["satellite"]["recent"][0]["outcome"] == "feedback -2"
    assert state["bursts"] == 1


@pytest.mark.parametrize("changes", [{"uplink": {"fail_next": "x"}}, {"uplink": {"voar": 1}},
                                     {"uplink": {"tx_enabled": "sim"}}])
def test_painel_recusa_pedido_invalido_sem_mudar_nada(changes):
    uplink, controller, _ = station()
    with pytest.raises(ValueError):
        controller.apply(changes)
    assert uplink.tx_enabled is True


# --- emissor: a resposta sai na próxima rajada ------------------------------------------


def fs2_cycle(payload: bytes):
    return em.fs2_beacon(NOMINAL, RATE, BAUD, bytes(16)).waveform_for(payload)


def test_resposta_passa_na_frente_e_a_forma_de_onda_base_volta():
    fs2 = em.fs2_beacon(NOMINAL, RATE, BAUD, bytes(16))
    base = fs2.waveform.copy()
    response = fs2_frames.ngham_frame(b"\x02 PY0EFS" + STATION)
    fs2.enqueue(response)

    first = fs2.take(len(fs2_cycle(response)))
    second = fs2.take(len(base))

    assert np.array_equal(first, fs2_cycle(response))
    assert np.array_equal(second, base)
    assert (fs2.responses_sent, fs2.queued_responses) == (1, 0)


def test_no_modo_manual_a_resposta_sai_sem_pedido():
    fs2 = em.fs2_beacon(NOMINAL, RATE, BAUD, bytes(16))
    fs2.continuous = False
    response = fs2_frames.ngham_frame(b"\x02 PY0EFS" + STATION)

    assert not fs2.take(1000).any()
    fs2.enqueue(response)
    assert fs2.take(1000).any()


# --- pela rede: o mesmo REQ que o grs-uplink faz -----------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_txserver_responde_pela_rede():
    uplink, _, sent = station()
    stop = threading.Event()
    address = f"tcp://127.0.0.1:{free_port()}"
    server = TxServer(address, uplink, stop)
    server.start()
    context = zmq.Context()
    request = context.socket(zmq.REQ)
    request.setsockopt(zmq.LINGER, 0)
    request.connect(address)
    try:
        header, iq = burst(ping(), burst_id="42")
        request.send_multipart([json.dumps(header).encode(), iq])
        assert request.poll(5000)
        reply = json.loads(request.recv())
    finally:
        stop.set()
        request.close()
        server.join(timeout=3)
        context.term()

    assert (reply["burst_id"], reply["status"]) == ("42", "ack")
    assert [radio for radio, _ in sent] == ["vhf"]
