"""Testes do Doppler da órbita real.

O simulador vira instrumento de medida da correção de Doppler da estação: se
o Doppler que ele impõe estiver errado, a estação "corrige" o erro dele e o
teste fica verde por engano. Por isso aqui se confere contra referências de
fora do código: um exemplo publicado de GMST, a forma do elipsoide, um
terceiro cálculo da velocidade radial (pela fórmula da velocidade, que o
módulo evita de propósito) e a física da passagem — aproximando, a frequência
sobe.

Nada vai à rede: o TLE é sintético, com época de hoje (ver `iss_like_tle`).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import jday

from sdr_sim import emitters as em
from sdr_sim import orbit
from sdr_sim.control import SimController
from sdr_sim.main import build_spectrum, parse_args
from sdr_sim.spectrum import VirtualSpectrum
from sdr_sim.station_tuning import StationTuningMonitor

SAMPLE_RATE = 240_000
CARRIER = 145_900_000.0
STATION = orbit.GroundStation(-23.5505, -46.6333, 760.0, "São Paulo")


def _checksum(line: str) -> str:
    return str(sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line) % 10)


def iss_like_tle(norad: int = 99001) -> tuple[str, str]:
    """Órbita tipo ISS (51,6°, ~15,5 rev/dia), que passa sobre São Paulo
    várias vezes por dia. Época de hoje e sem arrasto: nada decai entre a
    época e o teste."""
    now = datetime.now(timezone.utc)
    day = now.timetuple().tm_yday + (now.hour + now.minute / 60) / 24
    line1 = f"1 {norad:05d}U 98067A   {now:%y}{day:012.8f}  .00000000  00000-0  00000-0 0  999"
    line2 = f"2 {norad:05d}  51.6416 247.4627 0006703 130.5360 325.0288 15.49815645 4320"
    return line1 + _checksum(line1), line2 + _checksum(line2)


@pytest.fixture(scope="module")
def observer() -> orbit.Observer:
    return orbit.Observer(orbit.Satellite(*iss_like_tle(), name="ISS-SIM"), STATION)


@pytest.fixture(scope="module")
def a_pass(observer) -> orbit.Pass:
    return orbit.next_pass(observer, datetime.now(timezone.utc), min_peak_deg=20.0)


# --- referências externas --------------------------------------------------------


def test_gmst_bate_com_o_exemplo_do_vallado():
    """Vallado, Exemplo 3-5: 20/08/1992 12:14 UT1 -> GMST 152,578787886°."""
    jd, fr = jday(1992, 8, 20, 12, 14, 0)

    assert math.degrees(orbit.gmst_rad(jd + fr)) == pytest.approx(152.578787886, abs=1e-6)


def test_ecef_da_estacao_segue_o_wgs84():
    assert orbit.geodetic_to_ecef_km(0.0, 0.0, 0.0) == pytest.approx([6378.137, 0.0, 0.0])
    # Polo: o semieixo menor, b = a(1 - f).
    assert orbit.geodetic_to_ecef_km(90.0, 0.0, 0.0)[2] == pytest.approx(6356.7523142, abs=1e-6)


# --- TLE ---------------------------------------------------------------------------


def test_tle_valido_e_aceito():
    satellite = orbit.Satellite(*iss_like_tle(12345), name="  Alfa ")

    assert satellite.norad_id == 12345
    assert satellite.name == "Alfa"


@pytest.mark.parametrize("damage, message", [
    (lambda l1, l2: (l1[:-1] + str((int(l1[-1]) + 1) % 10), l2), "checksum"),
    (lambda l1, l2: (l2, l1), "começar com"),
    (lambda l1, l2: (l1[:40], l2), "69 caracteres"),
])
def test_tle_estragado_e_recusado_com_motivo(damage, message):
    """O TLE chega colado pelo operador: uma linha cortada pela metade tem de
    virar mensagem, e não uma órbita silenciosamente errada."""
    with pytest.raises(orbit.OrbitError, match=message):
        orbit.Satellite(*damage(*iss_like_tle()))


def test_linhas_de_satelites_diferentes_sao_recusadas():
    line1, _ = iss_like_tle(11111)
    _, line2 = iss_like_tle(22222)

    with pytest.raises(orbit.OrbitError, match="satélites diferentes"):
        orbit.Satellite(line1, line2)


OLD_ISS = [
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
]


def test_tle_velho_e_recusado_porque_o_sgp4_nao_recusa():
    """O TLE-exemplo clássico da ISS (2008). O SGP4 o propaga até hoje SEM
    erro e devolve uma posição inventada — foi este teste que mostrou isso.
    Quem recusa é a checagem de idade."""
    old = orbit.Satellite(*OLD_ISS)
    now = datetime.now(timezone.utc)

    old.position_ecef_km(now)  # não levanta: é exatamente o problema
    assert old.epoch.year == 2008
    with pytest.raises(orbit.OrbitError, match="velho demais"):
        old.check_age(now)


def test_painel_recusa_tle_velho():
    sim = controller()

    with pytest.raises(ValueError, match="velho demais"):
        sim.apply({"orbit": {"tle": OLD_ISS}})
    assert sim.emitter("fs2").doppler is None


# --- geometria da passagem ------------------------------------------------------------


def _range_rate_by_velocity(observer: orbit.Observer, when: datetime) -> float:
    """Terceiro cálculo: a fórmula da velocidade, com a rotação da Terra.
    É o caminho que o orbit.py evita — e o que a spacelab-tracking usa."""
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    _, r, v = observer.satellite._satrec.sgp4(jd, fr)
    theta = orbit.gmst_rad(jd + fr)
    c, s = math.cos(theta), math.sin(theta)
    r_ecef = np.array([c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]])
    v_rot = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
    omega = 7.2921158553e-5
    v_ecef = v_rot - np.cross([0.0, 0.0, omega], r_ecef)
    rho = r_ecef - orbit.geodetic_to_ecef_km(
        STATION.latitude_deg, STATION.longitude_deg, STATION.altitude_m)
    return float(np.dot(rho, v_ecef) / np.linalg.norm(rho))


def test_velocidade_radial_bate_com_a_formula_da_velocidade(observer, a_pass):
    """Diferença finita e fórmula da velocidade são contas independentes. Se
    concordam em 1 m/s (~0,5 Hz em 145,9 MHz), nenhuma das duas esqueceu a
    rotação da Terra (~0,4 km/s, que valeria centenas de hertz)."""
    for seconds in (0, 60, 180, 300):
        when = a_pass.aos + timedelta(seconds=seconds)
        assert observer.look(when).range_rate_km_s == pytest.approx(
            _range_rate_by_velocity(observer, when), abs=1e-3)


def test_aproximando_a_frequencia_sobe_e_afastando_desce(observer, a_pass):
    """O sinal que o simulador impõe. Invertido, a estação 'corrigiria' para
    o lado errado e dobraria o desvio — e um simulador com o mesmo erro faria
    o teste passar."""
    at_aos = observer.look(a_pass.aos)
    at_los = observer.look(a_pass.los)

    assert at_aos.range_rate_km_s < 0 < at_aos.doppler_hz(CARRIER)
    assert at_los.range_rate_km_s > 0 > at_los.doppler_hz(CARRIER)


def test_doppler_cruza_zero_na_culminacao_com_magnitude_de_leo(observer, a_pass):
    peak = observer.look(a_pass.peak)
    aos = observer.look(a_pass.aos)

    # Na culminação a distância para de cair: Doppler ~0.
    assert abs(peak.doppler_hz(CARRIER)) < 50.0
    # LEO a ~7,6 km/s em VHF: poucos kHz, nunca mais que v/c da órbita.
    assert 2_000.0 < aos.doppler_hz(CARRIER) < CARRIER * 7.9 / orbit.SPEED_OF_LIGHT_KM_S


def test_passagem_tem_forma_de_passagem(observer, a_pass):
    duration = (a_pass.los - a_pass.aos).total_seconds()

    assert 120 < duration < 16 * 60
    assert a_pass.aos < a_pass.peak < a_pass.los
    assert a_pass.peak_elevation_deg >= 20.0
    assert observer.elevation_deg(a_pass.aos - timedelta(seconds=30)) < 0.0
    assert observer.elevation_deg(a_pass.peak) == pytest.approx(a_pass.peak_elevation_deg)
    assert observer.elevation_deg(a_pass.los + timedelta(seconds=30)) < 0.0


def test_proxima_passagem_pula_a_que_ja_esta_em_andamento(observer, a_pass):
    """Uma passagem em curso não tem AOS para simular."""
    following = orbit.next_pass(observer, a_pass.peak, min_peak_deg=0.0)

    assert following.aos > a_pass.los


def test_satelite_que_nunca_passa_e_erro_com_mensagem():
    """Órbita equatorial vista do polo sul: nunca sobe."""
    now = datetime.now(timezone.utc)
    day = now.timetuple().tm_yday + (now.hour + now.minute / 60) / 24
    line1 = f"1 99002U 98067A   {now:%y}{day:012.8f}  .00000000  00000-0  00000-0 0  999"
    line2 = "2 99002   0.0100 247.4627 0006703 130.5360 325.0288 15.49815645 4320"
    satellite = orbit.Satellite(line1 + _checksum(line1), line2 + _checksum(line2))
    pole = orbit.Observer(satellite, orbit.GroundStation(-89.0, 0.0, 0.0))

    with pytest.raises(orbit.OrbitError, match="nenhuma passagem"):
        orbit.next_pass(pole, now, search_hours=6.0)


# --- o Doppler no sinal -------------------------------------------------------------------


def test_relogio_do_satelite_anda_com_o_tempo_simulado(observer, a_pass):
    doppler = orbit.OrbitalDoppler(observer, a_pass.aos, start_s=100.0)

    assert doppler.utc_at(100.0) == a_pass.aos
    assert doppler.utc_at(160.0) == a_pass.aos + timedelta(seconds=60)


def test_abaixo_do_horizonte_o_satelite_nao_e_ouvido(observer, a_pass):
    before = a_pass.aos - timedelta(minutes=5)
    with_horizon = orbit.OrbitalDoppler(observer, before, start_s=0.0, horizon=True)
    without = orbit.OrbitalDoppler(observer, before, start_s=0.0, horizon=False)

    assert not with_horizon.audible_at(0.0)
    assert with_horizon.audible_at(6 * 60.0)  # um minuto depois do AOS
    assert without.audible_at(0.0)


def test_satelite_abaixo_do_horizonte_some_do_espectro(observer, a_pass):
    beacon = em.carrier(CARRIER, SAMPLE_RATE)
    beacon.doppler = orbit.OrbitalDoppler(
        observer, a_pass.aos - timedelta(minutes=5), start_s=0.0, horizon=True)
    spectrum = VirtualSpectrum(CARRIER, SAMPLE_RATE, [beacon], snr_db=None)

    assert spectrum.visible() == []
    assert "abaixo do horizonte" in spectrum.describe()[0]
    assert not np.any(spectrum.block(4096))


def test_sintonizar_em_portadora_mais_doppler_centraliza_o_sinal(observer, a_pass):
    """O critério de aceite da task de Doppler, em miniatura: o receptor
    sintonizado onde a estação deveria mandá-lo põe o satélite no centro."""
    when = a_pass.aos + timedelta(seconds=60)
    beacon = em.carrier(CARRIER, SAMPLE_RATE)
    beacon.doppler = orbit.OrbitalDoppler(observer, when, start_s=0.0)
    shift = observer.look(when).doppler_hz(CARRIER)

    fixed = VirtualSpectrum(CARRIER, SAMPLE_RATE, [beacon], snr_db=None)
    assert fixed.visible()[0][1] == pytest.approx(shift, abs=1.0)

    beacon.doppler._last = None
    corrected = VirtualSpectrum(CARRIER + shift, SAMPLE_RATE, [beacon], snr_db=None)
    assert corrected.visible()[0][1] == pytest.approx(0.0, abs=1.0)


# --- controle pelo painel ---------------------------------------------------------------


def controller() -> SimController:
    spectrum = build_spectrum(parse_args(["--emitters=fs2", "--snr-db=none", "--seed=1"]))
    sim = SimController(spectrum, 4096)
    sim.station = STATION
    return sim


def test_orbita_por_tle_em_tempo_real():
    sim = controller()

    sim.apply({"orbit": {"tle": ["ISS-SIM", *iss_like_tle()]}})

    doppler = sim.emitter("fs2").doppler
    assert doppler.kind == "orbit" and doppler.mode == "realtime"
    state = sim.state()["emitters"][0]["doppler"]
    assert state["satellite"] == "ISS-SIM"
    assert abs(state["time_offset_s"]) < 2.0
    assert abs(state["shift_hz"]) < CARRIER * 7.9 / orbit.SPEED_OF_LIGHT_KM_S


def test_orbita_por_norad_usa_o_catalogo():
    sim = controller()
    asked = []

    def fake_fetch(norad):
        asked.append(norad)
        return orbit.Satellite(*iss_like_tle(norad), name="DO CATÁLOGO", source="celestrak")

    sim.fetch_satellite = fake_fetch
    sim.apply({"orbit": {"norad_id": 25544}})

    assert asked == [25544]
    assert sim.state()["emitters"][0]["doppler"]["source"] == "celestrak"


def test_proxima_passagem_comeca_trinta_segundos_antes_do_aos():
    sim = controller()

    sim.apply({"orbit": {"tle": list(iss_like_tle()), "mode": "next_pass"}})

    doppler = sim.emitter("fs2").doppler
    now_sim = sim.spectrum.elapsed_s
    assert doppler.utc_at(now_sim) == doppler.simulated_pass.aos - timedelta(
        seconds=orbit.NEXT_PASS_LEAD_S)
    # Ainda abaixo do horizonte: o sinal nasce daqui a 30 s, como no céu.
    assert not sim.emitter("fs2").audible_at(now_sim)
    assert sim.state()["emitters"][0]["doppler"]["pass"]["peak_elevation_deg"] >= 10.0


def test_desligar_a_orbita():
    sim = controller()
    sim.apply({"orbit": {"tle": list(iss_like_tle())}})

    sim.apply({"orbit": None})

    assert sim.emitter("fs2").doppler is None


@pytest.mark.parametrize("request_, message", [
    ({"orbit": {"tle": list(iss_like_tle()), "norad_id": 1}}, "OU"),
    ({"orbit": {}}, "OU"),
    ({"orbit": {"norad_id": True}}, "inteiro"),
    ({"orbit": {"tle": ["só uma linha"]}}, "linha1"),
    ({"orbit": {"tle": list(iss_like_tle()), "mode": "ontem"}}, "mode"),
    ({"orbit": {"tle": list(iss_like_tle()), "horizon": "sim"}}, "horizon"),
    ({"orbit": {"tle": list(iss_like_tle())}, "doppler": {"max_shift_hz": 0}}, "Doppler só"),
])
def test_pedido_de_orbita_invalido_nao_muda_nada(request_, message):
    sim = controller()

    with pytest.raises(ValueError, match=message):
        sim.apply(request_)

    assert sim.emitter("fs2").doppler is None


def test_modelo_em_s_continua_funcionando_e_se_identifica():
    sim = controller()

    sim.apply({"doppler": {"max_shift_hz": 3500, "pass_duration_s": 60}})

    assert sim.state()["emitters"][0]["doppler"]["kind"] == "model"


# --- comparação com a estação ------------------------------------------------------------


class FakeTuning:
    def __init__(self, frequency_hz, doppler_hz, age_s):
        self._snapshot = {"source": "tcp://station-manager:5581", "frequency_hz": frequency_hz,
                          "doppler_hz": doppler_hz, "doppler_age_s": age_s}

    def snapshot(self):
        return dict(self._snapshot)


def test_comparacao_traz_o_doppler_da_estacao_para_a_portadora_do_simulador():
    """A estação anuncia para 145,8 MHz; o FS-2 simulado está em 145,9. O
    Doppler é proporcional à portadora: sem a escala, a diferença mostrada
    seria ~0,07% do desvio, culpa da conta e não da órbita."""
    sim = controller()
    sim.apply({"orbit": {"tle": list(iss_like_tle())}})
    shift = sim.state()["emitters"][0]["doppler"]["shift_hz"]
    station_doppler = shift * 145_800_000.0 / CARRIER
    sim.station_tuning = FakeTuning(145_800_000.0, station_doppler, age_s=0.5)

    tuning = sim.state()["station_tuning"]

    assert tuning["comparison"]["difference_hz"] == pytest.approx(0.0, abs=2.0)
    assert tuning["comparison"]["carrier_difference_hz"] == pytest.approx(100_000.0)


def test_anuncio_velho_nao_entra_na_comparacao():
    sim = controller()
    sim.apply({"orbit": {"tle": list(iss_like_tle())}})
    sim.station_tuning = FakeTuning(CARRIER, 1234.0, age_s=60.0)

    tuning = sim.state()["station_tuning"]

    assert tuning["fresh"] is False and tuning["comparison"] is None


def test_monitor_le_freq_e_doppler():
    import threading

    monitor = StationTuningMonitor("tcp://ninguem:5581", threading.Event())
    monitor.handle([b"freq", b"145800000"])
    monitor.handle([b"doppler", b"-2345"])
    monitor.handle([b"doppler", b"lixo"])

    snapshot = monitor.snapshot()
    assert snapshot["frequency_hz"] == 145_800_000.0
    assert snapshot["doppler_hz"] == -2345.0
    assert snapshot["doppler_age_s"] < 1.0


def test_orbita_pela_linha_de_comando_aceita_vazio():
    """O compose passa --orbit-norad= quando SIM_ORBIT_NORAD não está definida."""
    assert parse_args(["--orbit-norad="]).orbit_norad is None
    assert parse_args(["--orbit-norad=25544"]).orbit_norad == 25544


def test_proxima_passagem_nao_e_comparada_com_a_estacao():
    """Relógio adiantado horas: comparar com o agora da estação daria
    quilohertz de 'diferença' que não é erro de ninguém."""
    sim = controller()
    sim.apply({"orbit": {"tle": list(iss_like_tle()), "mode": "next_pass"}})
    sim.station_tuning = FakeTuning(CARRIER, 1234.0, age_s=0.5)

    tuning = sim.state()["station_tuning"]

    assert tuning["fresh"] is True and tuning["comparison"] is None



def test_monitor_com_canal_so_le_o_seu_radio():
    """O SUB filtra por prefixo: sem comparar o tópico inteiro, o simulador
    do VHF compararia o próprio Doppler com o do UHF (3,2 vezes maior)."""
    import threading

    monitor = StationTuningMonitor("tcp://ninguem:5581", threading.Event(), channel="vhf")
    monitor.handle([b"freq.uhf", b"468400000"])
    monitor.handle([b"doppler.uhf", b"-7500"])
    monitor.handle([b"freq.vhf", b"145900000"])
    monitor.handle([b"doppler.vhf", b"-2345"])
    monitor.handle([b"doppler", b"1"])

    snapshot = monitor.snapshot()
    assert (snapshot["frequency_hz"], snapshot["doppler_hz"]) == (145_900_000.0, -2345.0)
    assert snapshot["channel"] == "vhf"
