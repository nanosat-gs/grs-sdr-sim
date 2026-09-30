"""Controle do simulador em execução: o que o painel lê e o que ele pode mudar.

Tudo o que o painel faz passa por aqui, sob a mesma trava que o laço de
geração segura enquanto produz um bloco. Sem ela, um retune no meio de
`spectrum.block()` produziria um bloco com metade das amostras numa sintonia
e metade noutra — um defeito que o demodulador leria como símbolo errado e
que ninguém conseguiria reproduzir depois.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from sdr_sim import orbit
from sdr_sim.emitters import PassDoppler
from sdr_sim.spectrum import VirtualSpectrum

SPECTRUM_BINS = 1024

# Limites do que o painel aceita. Não são limites físicos: são a faixa em que
# o resultado ainda significa alguma coisa para testar o cano.
SNR_RANGE_DB = (-20.0, 60.0)
AMPLITUDE_RANGE = (0.0, 2.0)
DOPPLER_MAX_HZ = 50_000.0
MAX_PACKETS_PER_REQUEST = 100

# Quanto tempo depois de começar a sair uma rajada ela pode ainda não ter
# chegado ao detector: a rajada (~0,17 s) + a janela do demodulador (0,5 s) +
# folga. Pacotes mais novos que isto não entram na conta de "perdidos" — senão
# o painel mostraria perda permanente só por causa do trânsito.
IN_FLIGHT_S = 1.5

# Anúncio de Doppler da estação mais velho que isto não entra na comparação: o
# Station Manager publica a cada tick (1 s), então parado = passagem acabou.
STATION_DOPPLER_FRESH_S = 5.0


class SimController:
    """Estado mutável do simulador, protegido por uma trava única."""

    def __init__(self, spectrum: VirtualSpectrum, block_samples: int) -> None:
        self.spectrum = spectrum
        self.block_samples = block_samples
        self.lock = threading.Lock()
        self.blocks = 0
        self.tune_source: str | None = None
        self.packets = None  # PacketMonitor, quando houver
        self.station_tuning = None  # StationTuningMonitor, quando houver
        # De onde o Doppler da órbita é calculado. As mesmas GS_* da estação.
        self.station = orbit.GroundStation.from_environment()
        # Trocável nos testes: é a única parte que vai à rede.
        self.fetch_satellite = orbit.fetch_celestrak
        self._last_block: np.ndarray | None = None
        # Instante simulado em que cada rajada do FS-2 começou a sair, e a
        # contagem de enviados no último "zerar".
        self._burst_starts: list[float] = []
        self._sent_baseline = 0

    # --- laço de geração -----------------------------------------------------

    def next_block(self) -> np.ndarray:
        with self.lock:
            fs2 = self._fs2()
            before = fs2.bursts_sent if fs2 else 0
            started_at = self.spectrum.elapsed_s

            block = self.spectrum.block(self.block_samples)
            self._last_block = block
            self.blocks += 1

            if fs2:
                self._burst_starts.extend([started_at] * (fs2.bursts_sent - before))
                # Só o trânsito recente importa; o resto é contagem.
                horizon = self.spectrum.elapsed_s - IN_FLIGHT_S
                while self._burst_starts and self._burst_starts[0] < horizon:
                    self._burst_starts.pop(0)

        return block

    def _fs2(self):
        try:
            return self.emitter("fs2")
        except ValueError:
            return None

    def _transmission(self) -> dict[str, Any] | None:
        """Enviados desde o último zerar, separando os que ainda podem estar
        a caminho. Chamado com a trava segura."""
        fs2 = self._fs2()
        if fs2 is None:
            return None

        sent = fs2.bursts_sent - self._sent_baseline
        in_flight = min(len(self._burst_starts), sent)

        return {
            "mode": "continuous" if fs2.continuous else "manual",
            "pending": fs2.pending,
            "sent": sent,
            "settled": sent - in_flight,
            "in_flight": in_flight,
        }

    # --- leitura -------------------------------------------------------------

    def emitter(self, name: str):
        for emitter in self.spectrum.emitters:
            if emitter.name == name:
                return emitter

        raise ValueError(f"emissor desconhecido: {name!r}")

    def fs2_period_s(self) -> float | None:
        """Um ciclo da rajada do FS-2 (quadro + silêncio), em segundos.

        É daqui que sai a taxa ESPERADA de pacotes. Comparar o que chega ao
        detector com o que o simulador de fato transmitiu é o que transforma
        "chegaram pacotes" em "chegaram X% dos pacotes".
        """
        try:
            fs2 = self.emitter("fs2")
        except ValueError:
            return None

        return len(fs2.waveform) / self.spectrum.sample_rate_hz

    def state(self) -> dict[str, Any]:
        with self.lock:
            spectrum = self.spectrum
            elapsed = spectrum.elapsed_s
            half_band = spectrum.sample_rate_hz / 2.0

            emitters = []
            for emitter in spectrum.emitters:
                offset = emitter.offset_from(spectrum.center_frequency_hz, elapsed)
                doppler = emitter.doppler
                emitters.append({
                    "name": emitter.name,
                    "frequency_hz": emitter.frequency_hz,
                    "enabled": emitter.enabled,
                    "amplitude": emitter.amplitude,
                    "offset_hz": offset,
                    "in_band": abs(offset) <= half_band,
                    "audible": emitter.audible_at(elapsed),
                    "doppler": _describe_doppler(doppler, elapsed, emitter.frequency_hz),
                })

            state = {
                "center_frequency_hz": spectrum.center_frequency_hz,
                "sample_rate_hz": spectrum.sample_rate_hz,
                "snr_db": spectrum.snr_db,
                "elapsed_s": elapsed,
                "blocks": self.blocks,
                "block_samples": self.block_samples,
                "retunes": spectrum.retune_count,
                "tune_source": self.tune_source,
                "emitters": emitters,
                "transmission": self._transmission(),
            }

        period = self.fs2_period_s()
        state["fs2_period_s"] = period
        state["packets"] = None if self.packets is None else self.packets.snapshot()
        state["station_tuning"] = self._station_comparison(state["emitters"])

        return state

    def _station_comparison(self, emitters: list[dict]) -> dict | None:
        """O Doppler que a estação anuncia ao lado do que o simulador impõe.

        A estação anuncia para a portadora cadastrada no TC Scheduler, que pode
        não ser a do FS-2 simulado. O Doppler é proporcional à portadora, então
        o número dela é trazido para a portadora do simulador antes de
        comparar. A diferença de portadora aparece à parte: é um offset que a
        malha de sintonia NÃO corrige, e o sinal sairia do centro por causa
        dela.
        """
        if self.station_tuning is None:
            return None

        snapshot = self.station_tuning.snapshot()
        fs2 = next((e for e in emitters if e["name"] == "fs2"), None)
        doppler = fs2["doppler"] if fs2 else None
        station_freq = snapshot["frequency_hz"]
        station_doppler = snapshot["doppler_hz"]
        age = snapshot["doppler_age_s"]
        fresh = age is not None and age <= STATION_DOPPLER_FRESH_S

        # Só em tempo real: na "próxima passagem" o relógio do satélite está
        # adiantado horas, e a estação calcula o agora — a diferença seria de
        # quilohertz e não diria nada sobre nenhuma das duas contas.
        comparison = None
        if (doppler and doppler["kind"] == "orbit" and doppler["mode"] == "realtime"
                and fresh and station_freq and station_doppler is not None):
            equivalent = station_doppler * fs2["frequency_hz"] / station_freq
            comparison = {
                "station_doppler_at_sim_carrier_hz": equivalent,
                "difference_hz": doppler["shift_hz"] - equivalent,
                "carrier_difference_hz": fs2["frequency_hz"] - station_freq,
            }

        return {**snapshot, "fresh": fresh, "comparison": comparison}

    def spectrum_db(self, bins: int = SPECTRUM_BINS) -> list[float]:
        """Espectro de potência do último bloco, em dB, do mais negativo ao mais positivo.

        Média de segmentos com janela de Hann, e não um FFT único: num FFT só,
        um espinho de ruído vence um sinal fraco de verdade e o gráfico pisca.
        """
        with self.lock:
            block = self._last_block

        if block is None:
            return []

        segments = len(block) // bins
        if segments == 0:
            return []

        frames = block[: segments * bins].reshape(segments, bins)
        window = np.hanning(bins).astype(np.float32)
        power = np.mean(np.abs(np.fft.fft(frames * window, axis=1)) ** 2, axis=0)
        power = np.fft.fftshift(power) / (bins * float(np.sum(window ** 2)))

        return np.round(10.0 * np.log10(power + 1e-20), 1).tolist()

    # --- escrita -------------------------------------------------------------

    def apply(self, changes: dict[str, Any]) -> None:
        """Aplica um pedido do painel. Valida TUDO antes de mudar qualquer coisa.

        Tudo-ou-nada de propósito: um pedido com um campo inválido não pode
        deixar o simulador meio alterado, porque o operador leria o painel
        achando que nada mudou.
        """
        if not isinstance(changes, dict):
            raise ValueError("esperava um objeto JSON")

        unknown = set(changes) - {
            "tune_hz", "snr_db", "emitters", "doppler", "orbit", "reset_packets",
            "fs2_mode", "send_packets",
        }
        if unknown:
            raise ValueError(f"campo desconhecido: {', '.join(sorted(unknown))}")

        actions = []

        if changes.get("reset_packets") is True:
            def reset_counts():
                fs2 = self._fs2()
                self._sent_baseline = fs2.bursts_sent if fs2 else 0
                self._burst_starts.clear()
                if self.packets is not None:
                    self.packets.reset()

            actions.append(reset_counts)

        if "fs2_mode" in changes or "send_packets" in changes:
            fs2 = self.emitter("fs2")
            mode = changes.get("fs2_mode", "continuous" if fs2.continuous else "manual")
            if mode not in ("continuous", "manual"):
                raise ValueError("fs2_mode precisa ser 'continuous' ou 'manual'")
            actions.append(_setter(fs2, "continuous", mode == "continuous"))

            if "send_packets" in changes:
                count = changes["send_packets"]
                if isinstance(count, bool) or not isinstance(count, int):
                    raise ValueError("send_packets precisa ser um inteiro")
                if not 1 <= count <= MAX_PACKETS_PER_REQUEST:
                    raise ValueError(
                        f"send_packets fora de [1, {MAX_PACKETS_PER_REQUEST}]"
                    )
                if mode != "manual":
                    raise ValueError(
                        "send_packets só vale no modo manual — no contínuo o FS-2 "
                        "já transmite sem parar"
                    )
                actions.append(lambda: fs2.trigger(count))

        if "tune_hz" in changes:
            tune_hz = _number(changes["tune_hz"], "tune_hz")
            if tune_hz <= 0:
                raise ValueError("tune_hz precisa ser positivo")
            actions.append(lambda: self.spectrum.tune(tune_hz))

        if "snr_db" in changes:
            raw = changes["snr_db"]
            if raw is None:
                snr = None
            else:
                snr = _number(raw, "snr_db")
                _check_range(snr, SNR_RANGE_DB, "snr_db")
            actions.append(lambda: setattr(self.spectrum, "snr_db", snr))

        if "emitters" in changes:
            requested = changes["emitters"]
            if not isinstance(requested, dict):
                raise ValueError("emitters precisa ser um objeto {nome: {...}}")

            for name, fields in requested.items():
                emitter = self.emitter(name)
                if not isinstance(fields, dict):
                    raise ValueError(f"emitters.{name} precisa ser um objeto")
                extra = set(fields) - {"enabled", "amplitude"}
                if extra:
                    raise ValueError(f"emitters.{name}: campo desconhecido {sorted(extra)}")

                if "enabled" in fields:
                    if not isinstance(fields["enabled"], bool):
                        raise ValueError(f"emitters.{name}.enabled precisa ser true/false")
                    actions.append(_setter(emitter, "enabled", fields["enabled"]))

                if "amplitude" in fields:
                    amplitude = _number(fields["amplitude"], f"emitters.{name}.amplitude")
                    _check_range(amplitude, AMPLITUDE_RANGE, f"emitters.{name}.amplitude")
                    actions.append(_setter(emitter, "amplitude", amplitude))

        if "doppler" in changes:
            doppler = changes["doppler"]
            if not isinstance(doppler, dict):
                raise ValueError("doppler precisa ser um objeto")
            fs2 = self.emitter("fs2")
            max_shift = _number(doppler.get("max_shift_hz", 0.0), "doppler.max_shift_hz")
            if abs(max_shift) > DOPPLER_MAX_HZ:
                raise ValueError(f"doppler.max_shift_hz fora de ±{DOPPLER_MAX_HZ:.0f}")

            if max_shift == 0:
                actions.append(_setter(fs2, "doppler", None))
            else:
                duration = _number(doppler.get("pass_duration_s", 600.0),
                                   "doppler.pass_duration_s")
                if duration <= 0:
                    raise ValueError("doppler.pass_duration_s precisa ser positivo")

                # A passagem começa AGORA, no tempo simulado — lido dentro da
                # trava, no momento de aplicar, e não na validação.
                def start_pass(fs2=fs2, max_shift=max_shift, duration=duration):
                    fs2.doppler = PassDoppler(max_shift, duration,
                                              start_s=self.spectrum.elapsed_s)

                actions.append(start_pass)

        if "orbit" in changes:
            if "doppler" in changes:
                raise ValueError("doppler e orbit no mesmo pedido: o FS-2 tem um Doppler só")
            fs2 = self.emitter("fs2")
            if changes["orbit"] is None:
                actions.append(_setter(fs2, "doppler", None))
            else:
                actions.append(self._orbit_action(fs2, changes["orbit"]))

        with self.lock:
            for action in actions:
                action()

    def _orbit_action(self, fs2, request: Any):
        """Valida e monta o Doppler da órbita FORA da trava.

        Buscar o TLE é rede e achar a próxima passagem é uma varredura de 48 h:
        segurar a trava durante isso congelaria o laço de geração, e o
        demodulador veria um buraco no IQ.
        """
        if not isinstance(request, dict):
            raise ValueError("orbit precisa ser um objeto")
        extra = set(request) - {"norad_id", "tle", "mode", "horizon", "min_peak_deg"}
        if extra:
            raise ValueError(f"orbit: campo desconhecido {sorted(extra)}")

        mode = request.get("mode", "realtime")
        if mode not in orbit.MODES:
            raise ValueError(f"orbit.mode precisa ser um de {', '.join(orbit.MODES)}")
        horizon = request.get("horizon", True)
        if not isinstance(horizon, bool):
            raise ValueError("orbit.horizon precisa ser true/false")

        if ("norad_id" in request) == ("tle" in request):
            raise ValueError("orbit precisa de norad_id OU tle")
        if "norad_id" in request:
            norad = request["norad_id"]
            if isinstance(norad, bool) or not isinstance(norad, int) or not 0 < norad < 10**6:
                raise ValueError("orbit.norad_id precisa ser um inteiro positivo")
            satellite = self.fetch_satellite(norad)
        else:
            lines = request["tle"]
            if (not isinstance(lines, list) or len(lines) not in (2, 3)
                    or not all(isinstance(line, str) for line in lines)):
                raise ValueError("orbit.tle precisa ser [linha1, linha2] ou [nome, linha1, linha2]")
            name = lines[0] if len(lines) == 3 else None
            satellite = orbit.Satellite(lines[-2], lines[-1], name)

        observer = orbit.Observer(satellite, self.station)
        now = datetime.now(timezone.utc)
        satellite.check_age(now)
        simulated_pass = None
        if mode == "next_pass":
            min_peak = _number(request.get("min_peak_deg", 10.0), "orbit.min_peak_deg")
            _check_range(min_peak, (0.0, 90.0), "orbit.min_peak_deg")
            simulated_pass = orbit.next_pass(observer, now, min_peak_deg=min_peak)
        else:
            # Propaga uma vez aqui: um TLE de satélite já reentrado falha no
            # pedido, com mensagem, e não no laço de geração.
            observer.look(now)

        def start():
            if simulated_pass is not None:
                # O relógio do satélite é adiantado: a passagem começa agora.
                utc_at_start = simulated_pass.aos - timedelta(seconds=orbit.NEXT_PASS_LEAD_S)
            else:
                utc_at_start = datetime.now(timezone.utc)
            fs2.doppler = orbit.OrbitalDoppler(
                observer, utc_at_start, start_s=self.spectrum.elapsed_s,
                mode=mode, horizon=horizon, simulated_pass=simulated_pass,
            )

        return start


def _describe_doppler(doppler, elapsed: float, carrier_hz: float) -> dict | None:
    if doppler is None:
        return None
    if doppler.kind == "orbit":
        return doppler.describe(elapsed, carrier_hz)
    return {
        "kind": "model",
        "max_shift_hz": doppler.max_shift_hz,
        "pass_duration_s": doppler.pass_duration_s,
        "progress_s": elapsed - doppler.start_s,
        "shift_hz": doppler.shift_at(elapsed),
    }


def _number(value: Any, field: str) -> float:
    # bool é subclasse de int em Python: sem esta exclusão, `true` passaria
    # como 1 Hz.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} precisa ser um número")
    if not math.isfinite(value):
        raise ValueError(f"{field} precisa ser finito")

    return float(value)


def _check_range(value: float, bounds: tuple[float, float], field: str) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{field} fora de [{low:g}, {high:g}]")


def _setter(obj: Any, attribute: str, value: Any):
    return lambda: setattr(obj, attribute, value)
