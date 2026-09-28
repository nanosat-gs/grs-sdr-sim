"""O espectro virtual: soma os emissores na banda-base do receptor.

É aqui que "estar sintonizado" passa a significar alguma coisa. Cada emissor
vive numa frequência absoluta; o receptor tem a sua; o que sai é a diferença.
Sintonize longe e o sinal some — que é exatamente o comportamento que se quer
poder exercitar antes de haver antena.
"""

from __future__ import annotations

import math

import numpy as np

from sdr_sim import modulation
from sdr_sim.emitters import Emitter


class VirtualSpectrum:
    """Receptor virtual: sintonia, taxa, ruído, e os emissores no ar."""

    def __init__(
        self,
        center_frequency_hz: float,
        sample_rate_hz: int,
        emitters: list[Emitter],
        snr_db: float | None = 20.0,
        seed: int | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("a taxa de amostragem precisa ser positiva")
        if center_frequency_hz <= 0:
            raise ValueError("a frequência de sintonia precisa ser positiva")

        self.center_frequency_hz = float(center_frequency_hz)
        self.sample_rate_hz = int(sample_rate_hz)
        self.emitters = emitters
        self.snr_db = snr_db

        self._rng = np.random.default_rng(seed)
        self._samples_emitted = 0
        self._retunes = 0

    @property
    def elapsed_s(self) -> float:
        """Tempo simulado, contado em amostras e não pelo relógio de parede.

        Deliberado: se a máquina engasgar, o sinal não deve pular no tempo. O
        Doppler tem de acompanhar as amostras que de fato saíram, ou o receptor
        veria um salto de frequência que nenhum satélite faria.
        """
        return self._samples_emitted / self.sample_rate_hz

    @property
    def retune_count(self) -> int:
        return self._retunes

    def tune(self, frequency_hz: float) -> None:
        """Muda a sintonia do receptor virtual."""
        if frequency_hz <= 0:
            raise ValueError(f"frequência inválida: {frequency_hz}")

        self.center_frequency_hz = float(frequency_hz)
        self._retunes += 1

    def visible(self) -> list[tuple[Emitter, float]]:
        """Emissores dentro da banda agora, com o offset de cada um.

        Fora de ±taxa/2 o emissor simplesmente não é ouvido. Um receptor de
        verdade faria o sinal *dobrar* de volta para dentro da banda (aliasing),
        e simular isso seria mais fiel — mas produziria um sinal fantasma no
        lugar errado, e alguém gastaria uma tarde caçando um bug que o
        simulador inventou. Silêncio é a mentira menos perigosa, e está
        documentada aqui.
        """
        half_band = self.sample_rate_hz / 2.0
        out = []

        for emitter in self.emitters:
            if not emitter.enabled:
                continue
            offset = emitter.offset_from(self.center_frequency_hz, self.elapsed_s)
            if abs(offset) <= half_band:
                out.append((emitter, offset))

        return out

    def block(self, n: int) -> np.ndarray:
        """Próximas `n` amostras de banda-base, com tudo somado."""
        total = np.zeros(n, dtype=np.complex64)

        audible = self.visible()
        heard = {id(emitter) for emitter, _ in audible}

        for emitter, offset in audible:
            total += emitter.mix(n, offset, self.sample_rate_hz)

        # Emissores fora da banda ainda consomem a forma de onda deles, para
        # que a cadência não congele enquanto o receptor está sintonizado
        # noutro lugar — o satélite não para de transmitir só porque ninguém
        # está ouvindo.
        for emitter in self.emitters:
            if id(emitter) not in heard:
                emitter.take(n)

        if self.snr_db is not None:
            total += self._noise_for(total, n)

        self._samples_emitted += n

        return total

    def _noise_for(self, signal: np.ndarray, n: int) -> np.ndarray:
        """Ruído dimensionado pela potência do sinal presente.

        Quando não há sinal nenhum (satélite calado, ou receptor sintonizado
        longe), usa-se uma referência fixa em vez de zero — senão o piso de
        ruído sumiria entre as rajadas, o que nenhum receptor faz e esconderia
        exatamente os bugs de detecção que aparecem no ruído.
        """
        power = float(np.mean(np.abs(signal) ** 2)) if n else 0.0
        reference = power if power > 1e-12 else 1.0

        return modulation.noise(n, self._rng) * math.sqrt(
            reference / (10.0 ** (self.snr_db / 10.0))
        )

    def describe(self) -> list[str]:
        """Uma linha por emissor, dizendo se está sendo ouvido e onde."""
        half_band = self.sample_rate_hz / 2.0
        lines = []

        for emitter in self.emitters:
            offset = emitter.offset_from(self.center_frequency_hz, self.elapsed_s)
            if not emitter.enabled:
                status = "desligado"
            elif abs(offset) <= half_band:
                status = "na banda"
            else:
                status = "FORA DA BANDA"
            lines.append(
                f"  {emitter.name:<10} {emitter.frequency_hz / 1e6:12.4f} MHz  "
                f"offset {offset / 1e3:+9.2f} kHz  {status}"
            )

        return lines
