"""Doppler da órbita real: o simulador passa a ser o satélite.

O `PassDoppler` é um modelo (curva em S). Este módulo põe no sinal o desvio que
o satélite de verdade teria, visto da estação: mesma órbita (TLE) e mesmas
coordenadas (`GS_*`) que o Station Manager usa.

A geometria é escrita aqui, e não importada da `spacelab-tracking`, pelo mesmo
motivo de a modulação não vir do demodulador. O Station Manager usa a
spacelab-tracking para CORRIGIR o Doppler; se o simulador a usasse para IMPOR
o Doppler, um erro nela (sinal trocado, rotação da Terra esquecida) apareceria
dos dois lados e se cancelaria — o espectro ficaria centrado e o teste,
verde, com a estação errada. Por isso a velocidade radial sai aqui de
**diferença finita entre duas distâncias**, sem velocidade nem ω×r: é o jeito
mais difícil de errar do mesmo jeito que a spacelab-tracking, que usa a
fórmula da velocidade.

Só o propagador (SGP4) é o mesmo, e de propósito: é ele que define o que um
TLE significa. Dois propagadores diferentes discordariam por metros, e a
diferença viraria ruído na comparação.

Tempo: o instante do satélite é `utc_at_start + (tempo simulado - start_s)`.
Como o tempo simulado corre pelas amostras (ver `VirtualSpectrum.elapsed_s`),
um engasgo da máquina atrasa o satélite junto com o sinal, em vez de fazê-lo
saltar.
"""

from __future__ import annotations

import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, jday

SPEED_OF_LIGHT_KM_S = 299_792.458

# WGS-84: é nele que as coordenadas GPS da estação estão.
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563

# Meio intervalo da diferença finita. Pequeno o bastante para a curvatura da
# distância não pesar (erro ~ dt² · aceleração radial ≈ milímetros/s), grande
# o bastante para os metros de ruído numérico do SGP4 não pesarem.
RANGE_RATE_HALF_STEP_S = 0.5

# Quanto antes do AOS a "próxima passagem" começa: o bastante para ver o sinal
# nascer no espectro, e não já no meio da subida.
NEXT_PASS_LEAD_S = 30.0

# Idade máxima do TLE em relação ao instante simulado. O SGP4 não recusa um
# TLE velho: propaga um de 2008 até hoje e devolve uma posição qualquer, sem
# erro. Em LEO o erro cresce ~km por dia; com 30 dias o Doppler ainda tem a
# forma certa, e passar disso é quase sempre TLE colado errado.
MAX_TLE_AGE_DAYS = 30.0

CELESTRAK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
# O CelesTrak recusa quem baixa o mesmo elemento várias vezes em pouco tempo.
# O TLE muda poucas vezes por dia; guardar duas horas não custa nada.
CELESTRAK_CACHE_S = 2 * 3600.0


class OrbitError(ValueError):
    """Órbita que não dá para usar: TLE ilegível, satélite que não existe no
    catálogo, SGP4 que diverge. ValueError para o painel devolvê-lo como 400."""


# --- estação ------------------------------------------------------------------


@dataclass(frozen=True)
class GroundStation:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    name: str = "estação"

    @classmethod
    def from_environment(cls) -> "GroundStation":
        """As mesmas variáveis `GS_*` que o compose repassa ao TC Scheduler e ao
        Station Manager (âncora `x-ground-station`). Os defaults são os do
        compose: se o simulador e a estação divergissem aqui, os dois
        calculariam o Doppler de lugares diferentes, sem erro nenhum."""
        return cls(
            latitude_deg=float(os.getenv("GS_LATITUDE_DEG", "-23.5505")),
            longitude_deg=float(os.getenv("GS_LONGITUDE_DEG", "-46.6333")),
            altitude_m=float(os.getenv("GS_ALTITUDE_M", "760")),
            name=os.getenv("GS_NAME", "estação"),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
        }


def geodetic_to_ecef_km(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    h = altitude_m / 1000.0
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_KM / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)

    return np.array([
        (n + h) * math.cos(lat) * math.cos(lon),
        (n + h) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - e2) + h) * math.sin(lat),
    ])


def gmst_rad(jd_ut1: float) -> float:
    """Tempo sideral médio de Greenwich (IAU-82), em radianos.

    Vallado, "Fundamentals of Astrodynamics", eq. 3-47, em segundos de tempo.
    Usa UTC no lugar de UT1: a diferença (< 0,9 s) gira a Terra ~0,004°, que
    muda o Doppler em frações de hertz.
    """
    t = (jd_ut1 - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    # 86400 s de tempo sideral = 360°, ou seja 240 s por grau.
    return math.radians((seconds % 86400.0) / 240.0)


# --- satélite ------------------------------------------------------------------


def _tle_checksum_ok(line: str) -> bool:
    digits = sum(int(c) for c in line[:68] if c.isdigit())
    minus = line[:68].count("-")
    return (digits + minus) % 10 == int(line[68])


def _check_tle(line1: str, line2: str) -> None:
    for number, line in ((1, line1), (2, line2)):
        if not isinstance(line, str) or len(line.rstrip()) < 69:
            raise OrbitError(f"linha {number} do TLE precisa ter 69 caracteres")
        if not line.startswith(f"{number} "):
            raise OrbitError(f"linha {number} do TLE precisa começar com '{number} '")
        if not line[68].isdigit() or not _tle_checksum_ok(line):
            raise OrbitError(f"linha {number} do TLE com checksum errado — copiada pela metade?")
    if line1[2:7] != line2[2:7]:
        raise OrbitError("as duas linhas do TLE são de satélites diferentes")


class Satellite:
    """Um TLE propagado pelo SGP4, com posição em ECEF."""

    def __init__(self, line1: str, line2: str, name: str | None = None, source: str = "tle") -> None:
        line1, line2 = line1.strip(), line2.strip()
        _check_tle(line1, line2)
        self.line1, self.line2 = line1, line2
        self.norad_id = int(line1[2:7])
        self.name = (name or "").strip() or f"NORAD {self.norad_id}"
        self.source = source
        self._satrec = Satrec.twoline2rv(line1, line2)
        self.epoch = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
            days=self._satrec.jdsatepoch + self._satrec.jdsatepochF - 2451545.0)

    def check_age(self, when: datetime) -> None:
        """Recusa propagar para longe demais da época (ver MAX_TLE_AGE_DAYS)."""
        age_days = (when - self.epoch).total_seconds() / 86400.0
        if abs(age_days) > MAX_TLE_AGE_DAYS:
            raise OrbitError(
                f"TLE de {self.name} tem época {self.epoch:%Y-%m-%d}, {abs(age_days):.0f} dias "
                f"do instante simulado — velho demais para o Doppler (máx. "
                f"{MAX_TLE_AGE_DAYS:.0f}). Pegue um TLE novo."
            )

    def position_ecef_km(self, when: datetime) -> np.ndarray:
        """TEME -> ECEF pela rotação de GMST (sem movimento do polo: metros)."""
        jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                      when.second + when.microsecond / 1e6)
        error, r, _velocity = self._satrec.sgp4(jd, fr)
        if error != 0:
            raise OrbitError(f"SGP4 falhou para {self.name}: {SGP4_ERRORS.get(error, error)}")

        theta = gmst_rad(jd + fr)
        c, s = math.cos(theta), math.sin(theta)
        x, y, z = r
        return np.array([c * x + s * y, -s * x + c * y, z])


_cache: dict[int, tuple[float, tuple[str, str, str]]] = {}
_cache_lock = threading.Lock()


def fetch_celestrak(norad_id: int, timeout_s: float = 10.0) -> Satellite:
    """O TLE mais recente do catálogo — a mesma fonte do TC Scheduler para
    satélites com NORAD ID. Chamado FORA da trava do simulador: é rede."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(norad_id)
    if cached and now - cached[0] < CELESTRAK_CACHE_S:
        name, line1, line2 = cached[1]
        return Satellite(line1, line2, name, source="celestrak")

    url = CELESTRAK_TLE_URL.format(norad=int(norad_id))
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise OrbitError(f"CelesTrak inacessível ({error}); cole o TLE à mão") from None

    lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    if len(lines) < 3 or not lines[1].startswith("1 "):
        # O CelesTrak responde 200 com "No GP data found" para ID inexistente.
        raise OrbitError(f"NORAD {norad_id} não encontrado no CelesTrak")

    name, line1, line2 = lines[0], lines[1], lines[2]
    satellite = Satellite(line1, line2, name, source="celestrak")
    with _cache_lock:
        _cache[norad_id] = (now, (name, line1, line2))
    return satellite


# --- geometria ------------------------------------------------------------------


@dataclass(frozen=True)
class Look:
    """O satélite visto da estação num instante."""

    when: datetime
    elevation_deg: float
    azimuth_deg: float
    range_km: float
    # Positivo quando o satélite se afasta.
    range_rate_km_s: float

    def doppler_hz(self, carrier_hz: float) -> float:
        """Desvio visto em solo. Aproximando (distância caindo) = positivo:
        a frequência recebida fica ACIMA da transmitida."""
        return -carrier_hz * self.range_rate_km_s / SPEED_OF_LIGHT_KM_S


class Observer:
    """Calcula o que a estação vê de um satélite."""

    def __init__(self, satellite: Satellite, station: GroundStation) -> None:
        self.satellite = satellite
        self.station = station
        self._station_ecef = geodetic_to_ecef_km(
            station.latitude_deg, station.longitude_deg, station.altitude_m
        )
        lat = math.radians(station.latitude_deg)
        lon = math.radians(station.longitude_deg)
        # Linhas da matriz ECEF -> ENU (leste, norte, zênite) da estação.
        self._enu = np.array([
            [-math.sin(lon), math.cos(lon), 0.0],
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        ])

    def _range_km(self, when: datetime) -> float:
        return float(np.linalg.norm(self.satellite.position_ecef_km(when) - self._station_ecef))

    def elevation_deg(self, when: datetime) -> float:
        rho = self.satellite.position_ecef_km(when) - self._station_ecef
        _east, _north, up = self._enu @ rho
        return math.degrees(math.asin(up / float(np.linalg.norm(rho))))

    def look(self, when: datetime) -> Look:
        rho = self.satellite.position_ecef_km(when) - self._station_ecef
        distance = float(np.linalg.norm(rho))
        east, north, up = self._enu @ rho

        step = timedelta(seconds=RANGE_RATE_HALF_STEP_S)
        range_rate = (self._range_km(when + step) - self._range_km(when - step)) / (
            2.0 * RANGE_RATE_HALF_STEP_S
        )

        return Look(
            when=when,
            elevation_deg=math.degrees(math.asin(up / distance)),
            azimuth_deg=math.degrees(math.atan2(east, north)) % 360.0,
            range_km=distance,
            range_rate_km_s=range_rate,
        )


@dataclass(frozen=True)
class Pass:
    aos: datetime
    los: datetime
    peak: datetime
    peak_elevation_deg: float

    def to_dict(self) -> dict:
        return {
            "aos": self.aos.isoformat(),
            "los": self.los.isoformat(),
            "peak": self.peak.isoformat(),
            "peak_elevation_deg": self.peak_elevation_deg,
            "duration_s": (self.los - self.aos).total_seconds(),
        }


def _refine_crossing(observer: Observer, below: datetime, above: datetime) -> datetime:
    """Bissecção até 1 s do instante em que a elevação cruza 0°."""
    while abs((above - below).total_seconds()) > 1.0:
        middle = below + (above - below) / 2
        if observer.elevation_deg(middle) >= 0.0:
            above = middle
        else:
            below = middle
    return above


def _refine_peak(observer: Observer, around: datetime, half_width_s: float) -> tuple[datetime, float]:
    """Busca ternária do pico em ±`half_width_s` do melhor ponto amostrado:
    perto da culminação a elevação tem um máximo só."""
    low = around - timedelta(seconds=half_width_s)
    high = around + timedelta(seconds=half_width_s)
    while (high - low).total_seconds() > 0.5:
        third = (high - low) / 3
        if observer.elevation_deg(low + third) < observer.elevation_deg(high - third):
            low = low + third
        else:
            high = high - third
    peak = low + (high - low) / 2
    return peak, observer.elevation_deg(peak)


def next_pass(
    observer: Observer,
    after: datetime,
    min_peak_deg: float = 10.0,
    search_hours: float = 48.0,
    step_s: float = 30.0,
) -> Pass:
    """A próxima passagem que COMEÇA depois de `after` e culmina acima de
    `min_peak_deg`. Uma passagem já em andamento em `after` é pulada: ela não
    teria AOS para simular."""
    step = timedelta(seconds=step_s)
    end = after + timedelta(hours=search_hours)

    t = after
    previous = observer.elevation_deg(t)
    while t < end:
        t_next = t + step
        current = observer.elevation_deg(t_next)
        if previous < 0.0 <= current:
            aos = _refine_crossing(observer, t, t_next)

            # Sobe até cair de novo, guardando o pico amostrado.
            peak_t, peak_el = t_next, current
            u = t_next
            while True:
                u_next = u + step
                el = observer.elevation_deg(u_next)
                if el > peak_el:
                    peak_t, peak_el = u_next, el
                if el < 0.0:
                    los = _refine_crossing(observer, u_next, u)
                    break
                u = u_next

            peak_t, peak_el = _refine_peak(observer, peak_t, step_s)
            if peak_el >= min_peak_deg:
                return Pass(aos=aos, los=los, peak=peak_t, peak_elevation_deg=peak_el)
            t, previous = los, observer.elevation_deg(los)
            continue

        t, previous = t_next, current

    raise OrbitError(
        f"nenhuma passagem de {observer.satellite.name} acima de {min_peak_deg:g}° "
        f"nas próximas {search_hours:g} h"
    )


# --- o Doppler que o emissor usa ---------------------------------------------------


MODES = ("realtime", "next_pass")


class OrbitalDoppler:
    """Mesmo papel do `PassDoppler` no emissor, com a geometria real.

    `horizon`: abaixo de 0° o satélite não é ouvido — como no céu. Desligado,
    ele é ouvido sempre, o que serve para testar a malha de Doppler na bancada
    com uma passagem sintética do `tools/station_demo.py`, que aponta para um
    satélite do outro lado da Terra.
    """

    kind = "orbit"

    def __init__(
        self,
        observer: Observer,
        utc_at_start: datetime,
        start_s: float,
        mode: str = "realtime",
        horizon: bool = True,
        simulated_pass: Pass | None = None,
    ) -> None:
        if mode not in MODES:
            raise OrbitError(f"modo desconhecido: {mode!r}")
        self.observer = observer
        self.utc_at_start = utc_at_start
        self.start_s = start_s
        self.mode = mode
        self.horizon = horizon
        self.simulated_pass = simulated_pass
        # O laço de geração e o painel pedem o mesmo instante várias vezes por
        # bloco; três SGP4 por pedido é barato, mas não de graça.
        self._last: tuple[float, Look] | None = None

    @property
    def satellite(self) -> Satellite:
        return self.observer.satellite

    def utc_at(self, elapsed_s: float) -> datetime:
        return self.utc_at_start + timedelta(seconds=elapsed_s - self.start_s)

    def look_at(self, elapsed_s: float) -> Look:
        if self._last is not None and self._last[0] == elapsed_s:
            return self._last[1]
        look = self.observer.look(self.utc_at(elapsed_s))
        self._last = (elapsed_s, look)
        return look

    def shift_at(self, elapsed_s: float, carrier_hz: float | None = None) -> float:
        if carrier_hz is None:
            raise ValueError("o Doppler da órbita depende da portadora")
        return self.look_at(elapsed_s).doppler_hz(carrier_hz)

    def audible_at(self, elapsed_s: float) -> bool:
        return not self.horizon or self.look_at(elapsed_s).elevation_deg >= 0.0

    def describe(self, elapsed_s: float, carrier_hz: float) -> dict:
        look = self.look_at(elapsed_s)
        simulated_now = self.utc_at(elapsed_s)
        return {
            "kind": self.kind,
            "satellite": self.satellite.name,
            "norad_id": self.satellite.norad_id,
            "source": self.satellite.source,
            "mode": self.mode,
            "horizon": self.horizon,
            "utc": simulated_now.isoformat(),
            # ~0 em tempo real. Crescendo devagar = a máquina não acompanha o
            # ritmo e o satélite simulado está ficando para trás do relógio.
            "time_offset_s": (simulated_now - datetime.now(timezone.utc)).total_seconds(),
            "elevation_deg": look.elevation_deg,
            "azimuth_deg": look.azimuth_deg,
            "range_km": look.range_km,
            "range_rate_km_s": look.range_rate_km_s,
            "shift_hz": look.doppler_hz(carrier_hz),
            "audible": self.audible_at(elapsed_s),
            "pass": None if self.simulated_pass is None else self.simulated_pass.to_dict(),
            "station": self.observer.station.to_dict(),
        }
