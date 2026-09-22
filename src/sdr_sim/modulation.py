"""Geração das formas de onda que o SDR virtual transmite.

Nota importante sobre independência: este módulo **não** usa o DSP do
`grs-demodulator`. Poderia — a classe GMSK de lá modula e demodula — e foi o
que a primeira bancada fez, por preguiça honesta. Mas um teste em que o mesmo
código modula e demodula prova menos do que parece: dois erros simétricos se
cancelam e o teste passa em verde com o sinal errado.

Aqui a modulação é escrita do zero, a partir da definição. Se o demodulador
recuperar estes bits, é porque os dois concordam sobre o que é 2GFSK — e não
porque compartilham a mesma convenção acidental.
"""

from __future__ import annotations

import math

import numpy as np


def gaussian_pulse(bt: float, samples_per_symbol: int, span_symbols: int = 4) -> np.ndarray:
    """Filtro de conformação gaussiano, normalizado para área unitária.

    `bt` é o produto banda×período de símbolo. Quanto menor, mais estreito o
    espectro e mais interferência entre símbolos — 0.5 é o valor que o
    demodulador da estação assume.
    """
    if bt <= 0:
        raise ValueError(f"BT precisa ser positivo, veio {bt}")

    t = np.arange(-span_symbols * samples_per_symbol, span_symbols * samples_per_symbol + 1)
    t = t / samples_per_symbol

    # sigma = sqrt(ln 2) / (2*pi*BT), com t em períodos de símbolo. É a
    # definição do pulso gaussiano do GMSK, e o valor importa muito mais do que
    # parece: um sigma largo demais espalha cada símbolo por vários vizinhos e
    # APAGA um preâmbulo alternado, que é o padrão mais rápido que existe. O
    # sintoma não é ruído — é o demodulador entregando bits enviesados, sem
    # syncword, como se o sinal fosse outro.
    sigma = math.sqrt(math.log(2.0)) / (2.0 * math.pi * bt)
    pulse = np.exp(-(t**2) / (2.0 * sigma**2))

    return pulse / pulse.sum()


def bits_to_symbols(bits: np.ndarray) -> np.ndarray:
    """0/1 -> -1/+1. NRZ, que é o que a portadora desvia."""
    return 2.0 * np.asarray(bits, dtype=np.float64) - 1.0


def modulate_2gfsk(
    bits: np.ndarray,
    samples_per_symbol: int,
    bt: float = 0.5,
    modulation_index: float = 0.5,
) -> np.ndarray:
    """Bits -> banda-base complexa 2GFSK.

    A definição, e nada além dela:

        símbolos NRZ -> sobreamostra -> filtro gaussiano -> integra a fase
        -> exp(j·fase)

    `modulation_index` (h) é o desvio de pico em unidades de meia taxa de
    símbolo: h = 0.5 é o caso GMSK, e é o que o demodulador da estação espera.
    O firmware do TTC2 (Si446x) declara 2GFSK, que é a mesma família — GMSK é
    2GFSK com h exatamente 0.5.

    :return: complex64 em amplitude unitária.
    """
    if samples_per_symbol < 2:
        raise ValueError(
            f"{samples_per_symbol} amostras por símbolo é pouco; abaixo de 2 não há"
            " sinal para recuperar tempo"
        )

    symbols = bits_to_symbols(bits)

    # Sobreamostragem por repetição (segura e explícita), depois conformação.
    upsampled = np.repeat(symbols, samples_per_symbol)
    shaped = np.convolve(upsampled, gaussian_pulse(bt, samples_per_symbol), mode="same")

    # Integra para virar fase. O fator abaixo é o que faz h significar o que
    # diz: o desvio de pico fica em h·(taxa de símbolo)/2.
    phase_step = np.pi * modulation_index / samples_per_symbol
    phase = np.cumsum(shaped) * phase_step

    return np.exp(1j * phase).astype(np.complex64)


def modulate_fm(
    audio: np.ndarray,
    sample_rate_hz: float,
    deviation_hz: float,
    preemphasis_us: float | None = 75.0,
) -> np.ndarray:
    """Áudio real em ±1 -> banda-base complexa FM.

    Existe para exercitar o caminho de áudio do gravador, e para dar à estação
    um sinal que uma pessoa reconhece de ouvido — o que é um teste melhor do
    que parece: desafinação, estouro e inversão aparecem no ouvido antes de
    aparecer num gráfico.

    A pré-ênfase é o espelho da de-ênfase do receptor: levanta os agudos antes
    de transmitir para que a de-ênfase do outro lado os baixe de volta junto
    com o ruído. Aplicá-la aqui é o que torna o par testável de ponta a ponta.
    """
    audio = np.asarray(audio, dtype=np.float64)

    if preemphasis_us:
        tau = preemphasis_us * 1e-6
        alpha = float(np.exp(-1.0 / (sample_rate_hz * tau)))
        # Diferenciador de um polo: o inverso exato do integrador da de-ênfase.
        emphasized = np.empty_like(audio)
        previous = 0.0
        for index, value in enumerate(audio):
            emphasized[index] = value - alpha * previous
            previous = value
        peak = np.max(np.abs(emphasized))
        audio = emphasized / peak if peak > 0 else emphasized

    phase = np.cumsum(audio) * (2.0 * np.pi * deviation_hz / sample_rate_hz)

    return np.exp(1j * phase).astype(np.complex64)


def tone(
    frequency_hz: float, duration_s: float, sample_rate_hz: float, amplitude: float = 1.0
) -> np.ndarray:
    """Tom senoidal real, para usar como áudio de entrada da FM."""
    n = int(round(duration_s * sample_rate_hz))
    t = np.arange(n) / sample_rate_hz

    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def melody(sample_rate_hz: float, note_duration_s: float = 0.35) -> np.ndarray:
    """Um arpejo curto, para o teste de ouvido.

    Notas reconhecíveis em vez de um tom só: um tom puro soa igual demodulado
    certo ou com a taxa errada por um fator pequeno, enquanto uma melodia
    desafinada é evidente na hora.
    """
    # Lá maior: A4, C#5, E5, A5.
    notes = (440.00, 554.37, 659.25, 880.00)

    return np.concatenate(
        [tone(frequency, note_duration_s, sample_rate_hz, 0.8) for frequency in notes]
    )


def noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ruído gaussiano complexo, com potência total unitária."""
    scale = np.sqrt(0.5)

    return (rng.normal(0.0, scale, n) + 1j * rng.normal(0.0, scale, n)).astype(np.complex64)
