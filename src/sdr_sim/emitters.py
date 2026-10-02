"""O que existe no ar, no espectro simulado.

Um emissor tem uma frequência ABSOLUTA e uma forma de onda. O receptor tem uma
frequência de sintonia. O que o receptor ouve é a diferença — e é essa
separação que torna a sintonia testável: apontar o receptor para o lugar
errado faz o sinal sair do centro e, se errar bastante, sumir da banda. Um
simulador que sempre entrega o sinal centrado não testaria sintonia nenhuma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sdr_sim import modulation

# Syncword do NGHam, direto da implementacao de referencia:
#
#     const uint8_t NGH_SYNC[] = {0x5D, 0xE6, 0x2A, 0x7E};   (ngham.c)
#
# NAO 0xBA 0x67 0x54 0x7E, que e o MESMO vetor com os bits de cada byte
# invertidos. O simulador gerava a versao invertida, e o detector procurava a
# versao invertida: os dois concordavam entre si e ambos discordavam do
# satelite. So uma gravacao real do FloripaSat-1 pegou.
NGHAM_SYNCWORD = bytes((0x5D, 0xE6, 0x2A, 0x7E))

# Preambulo do NGHam. 0xAA, e nao 0x55 -- tambem confirmado na gravacao real,
# onde os 16 bits antes do primeiro sync sao 1010101010101010.
NGHAM_PREAMBLE = 0xAA


def ccsds_pn(length: int) -> bytes:
    """A sequência pseudoaleatória do scrambler CCSDS (CCSDS 131.0-B):
    h(x) = x^8 + x^7 + x^5 + x^3 + 1, registrador todo em 1.

    Começa FF 48 0E C0 9A 0D 70 BC — igual, byte a byte, à tabela
    `ccsds_poly` do NGHam no firmware do TTC 2.0 (ccsds_scrambler.c).
    """
    state = [1] * 8
    out = bytearray()
    for _ in range(length):
        byte = 0
        for _ in range(8):
            byte = (byte << 1) | state[0]
            state = state[1:] + [state[0] ^ state[3] ^ state[5] ^ state[7]]
        out.append(byte)
    return bytes(out)


def ccsds_scramble(data: bytes) -> bytes:
    """XOR com a sequência CCSDS, como o NGHam faz com o codeword.

    Por que o simulador embaralha: o satélite embaralha, e isso deixa os bits
    EQUILIBRADOS (metade 0, metade 1). Sem isso o payload `00 01 02 ... 3F`
    tem 37,5% de uns, o tom do bit 0 fica mais forte que o do bit 1, e o
    centro do espectro escorrega ~50 Hz a 1200 baud — o bloco FFT mediu
    exatamente isso e o ajuste fino "corrigiu" um erro que o satélite real
    não teria.
    """
    return bytes(a ^ b for a, b in zip(data, ccsds_pn(len(data))))


def bytes_to_bits(data: bytes) -> np.ndarray:
    """MSB primeiro, que é a ordem em que o NGH_SYNC do ngham.c está escrito."""
    return np.array([(byte >> (7 - i)) & 1 for byte in data for i in range(8)], dtype=np.uint8)


def build_frame(payload: bytes, preamble_bytes: int = 32) -> np.ndarray:
    """Preâmbulo + syncword + payload, como fluxo de bits.

    O preâmbulo é 0xAA — alternado a cada bit. Não é enfeite: é o que dá ao
    sincronismo de tempo do receptor transições suficientes para travar ANTES
    de o syncword passar. Sem ele, o Mueller & Muller ainda está convergindo
    quando o syncword chega, e o detector não acha nada.
    """
    return bytes_to_bits(bytes([NGHAM_PREAMBLE] * preamble_bytes) + NGHAM_SYNCWORD + payload)


class PassDoppler:
    """Desvio Doppler ao longo de uma passagem.

    MODELO, não propagação orbital: uma curva em S que começa em +máximo
    (satélite se aproximando), cruza zero na aproximação máxima e termina em
    -máximo. A forma real depende da geometria da passagem, mas o que importa
    para testar o cano é a ordem de grandeza e o sentido da varredura — quem
    calcula Doppler de verdade é a spacelab_tracking, no Station Manager.

    Para o número de uma passagem real, use `orbit.OrbitalDoppler`: mesma
    interface, com a geometria do satélite de verdade.
    """

    kind = "model"

    def __init__(
        self, max_shift_hz: float, pass_duration_s: float, start_s: float = 0.0
    ) -> None:
        if pass_duration_s <= 0:
            raise ValueError("a duração da passagem precisa ser positiva")

        self.max_shift_hz = max_shift_hz
        self.pass_duration_s = pass_duration_s
        # Instante simulado em que a passagem começa. Uma passagem iniciada
        # pelo painel com o simulador já rodando começa "agora", e não no
        # instante zero — senão ela nasceria no meio da curva.
        self.start_s = start_s
        # Escolhido para a curva gastar a maior parte da varredura perto da
        # aproximação máxima, como numa passagem de verdade.
        self._steepness = 6.0 / pass_duration_s

    def shift_at(self, elapsed_s: float, carrier_hz: float | None = None) -> float:
        # `carrier_hz` é ignorado: o pico do modelo já é dado em Hz. Está na
        # assinatura porque o Doppler da órbita precisa dele.
        centered = (elapsed_s - self.start_s) - self.pass_duration_s / 2.0

        return -self.max_shift_hz * math.tanh(self._steepness * centered)

    def audible_at(self, elapsed_s: float) -> bool:
        # O modelo não tem horizonte: o satélite é ouvido a passagem toda.
        return True


@dataclass
class Emitter:
    """Uma fonte no espectro simulado.

    `waveform` é cíclico: o simulador consome dele indefinidamente, voltando ao
    início quando acaba. Para rajadas, o silêncio entre elas faz parte da forma
    de onda — assim o ciclo inteiro é uma cadência realista sem o simulador
    precisar de máquina de estados.
    """

    name: str
    frequency_hz: float
    waveform: np.ndarray
    amplitude: float = 1.0
    # PassDoppler (modelo) ou orbit.OrbitalDoppler (órbita real): os dois
    # respondem shift_at(elapsed, portadora) e audible_at(elapsed).
    doppler: PassDoppler | None = None
    # Erro do oscilador do transmissor: a portadora sai aqui, e não na
    # nominal. O TTC 2.0 declara cristal de ±10 ppm (até ±1,5 kHz em 145,9 MHz,
    # ±4,7 kHz em 468,4 MHz). É o que o Doppler previsto não vê e o ajuste
    # fino (bloco FFT) tem de achar.
    carrier_offset_hz: float = 0.0
    # Desligado é tratado como fora da banda: não é ouvido, mas continua
    # consumindo a forma de onda, para a cadência não congelar.
    enabled: bool = True
    # Contínuo: repete a forma de onda para sempre. Manual: silêncio até
    # `trigger()`, e então um ciclo (quadro + silêncio) por pedido.
    continuous: bool = True
    # Ciclos iniciados desde a partida. Para uma rajada, é quantos pacotes
    # foram de fato transmitidos — o denominador honesto de "quantos chegaram".
    bursts_sent: int = field(default=0, repr=False)
    _pending: int = field(default=0, repr=False)

    # Posição corrente no ciclo e fase acumulada do deslocamento em frequência.
    # A fase PRECISA atravessar blocos: reiniciá-la a cada bloco criaria um
    # salto de fase artificial a cada fronteira, que o demodulador leria como
    # símbolo errado — é o mesmo defeito de borda que a bancada expôs do outro
    # lado do cano.
    _cursor: int = field(default=0, repr=False)
    _phase: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if len(self.waveform) == 0:
            raise ValueError(f"emissor {self.name} sem forma de onda")
        self.waveform = self.waveform.astype(np.complex64)

    @property
    def pending(self) -> int:
        return self._pending

    def trigger(self, count: int = 1) -> None:
        """Enfileira `count` ciclos no modo manual."""
        if count <= 0:
            raise ValueError("count precisa ser positivo")
        self._pending += count

    def take(self, n: int) -> np.ndarray:
        """Próximas `n` amostras da forma de onda.

        Um ciclo começado sempre vai até o fim, mesmo que o modo mude no
        meio: cortar uma rajada ao meio transmitiria um pacote truncado, e
        ele seria contado como enviado e perdido sem que o cano tivesse culpa.
        """
        out = np.zeros(n, dtype=np.complex64)
        filled = 0

        while filled < n:
            if self._cursor == 0:
                if self.continuous:
                    self.bursts_sent += 1
                elif self._pending > 0:
                    self._pending -= 1
                    self.bursts_sent += 1
                else:
                    break  # manual e nada pedido: o resto do bloco é silêncio

            chunk = min(n - filled, len(self.waveform) - self._cursor)
            out[filled : filled + chunk] = self.waveform[self._cursor : self._cursor + chunk]
            filled += chunk
            self._cursor = (self._cursor + chunk) % len(self.waveform)

        return out

    def offset_from(self, center_hz: float, elapsed_s: float) -> float:
        """Onde este emissor cai na banda-base do receptor, em Hz."""
        actual = self.frequency_hz + self.carrier_offset_hz
        shift = self.doppler.shift_at(elapsed_s, actual) if self.doppler else 0.0

        return (actual + shift) - center_hz

    def audible_at(self, elapsed_s: float) -> bool:
        """Falso com o satélite abaixo do horizonte (Doppler de órbita)."""
        return self.doppler is None or self.doppler.audible_at(elapsed_s)

    def mix(self, n: int, offset_hz: float, sample_rate_hz: float) -> np.ndarray:
        """Desloca `n` amostras para `offset_hz`, mantendo a fase entre blocos."""
        samples = self.take(n)

        step = 2.0 * math.pi * offset_hz / sample_rate_hz
        phases = self._phase + step * np.arange(n, dtype=np.float64)
        # Guarda o ângulo já embrulhado: sem isso, uma execução longa acumula
        # um float enorme e a precisão do cosseno degrada de forma visível.
        self._phase = float((self._phase + step * n) % (2.0 * math.pi))

        return (samples * np.exp(1j * phases).astype(np.complex64)) * self.amplitude


def fs2_beacon(
    frequency_hz: float,
    sample_rate_hz: int,
    baud: int,
    payload: bytes,
    gap_s: float = 0.5,
    bt: float = 0.5,
    amplitude: float = 1.0,
    doppler: PassDoppler | None = None,
) -> Emitter:
    """Rajadas 2GFSK com enquadramento NGHam, como o downlink do FS-2."""
    if sample_rate_hz % baud != 0:
        raise ValueError(
            f"{sample_rate_hz} S/s não é múltiplo de {baud} baud; o resto vira erro de fase"
        )

    burst = modulation.modulate_2gfsk(
        build_frame(payload), samples_per_symbol=sample_rate_hz // baud, bt=bt
    )
    silence = np.zeros(int(round(gap_s * sample_rate_hz)), dtype=np.complex64)

    return Emitter(
        name="fs2",
        frequency_hz=frequency_hz,
        waveform=np.concatenate((burst, silence)),
        amplitude=amplitude,
        doppler=doppler,
    )


def fm_station(
    frequency_hz: float,
    sample_rate_hz: int,
    deviation_hz: float = 50_000.0,
    amplitude: float = 1.0,
    doppler: PassDoppler | None = None,
) -> Emitter:
    """Uma portadora FM com uma melodia, para o teste de ouvido."""
    audio = modulation.melody(sample_rate_hz)

    return Emitter(
        name="fm",
        frequency_hz=frequency_hz,
        waveform=modulation.modulate_fm(audio, sample_rate_hz, deviation_hz),
        amplitude=amplitude,
        doppler=doppler,
    )


def carrier(
    frequency_hz: float, sample_rate_hz: int, amplitude: float = 1.0
) -> Emitter:
    """Portadora contínua sem modulação — a referência mais simples que existe.

    Serve para conferir sintonia: se o receptor está no lugar certo, ela aparece
    exatamente no offset esperado do espectro. Qualquer erro de frequência é
    lido direto no gráfico, sem passar por demodulação.
    """
    return Emitter(
        name="carrier",
        frequency_hz=frequency_hz,
        waveform=np.ones(sample_rate_hz, dtype=np.complex64),
        amplitude=amplitude,
    )
