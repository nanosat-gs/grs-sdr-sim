"""Testes do SDR virtual.

Um simulador é instrumento de medida: se ele mentir, todo teste feito com ele
mente junto, e de forma convincente. Por isso o que se confere aqui não é que
"roda", e sim que o sinal tem as propriedades declaradas — a portadora cai no
offset prometido, o desvio 2GFSK é o pedido, o SNR é o configurado, e a fase
não salta entre blocos.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdr_sim import emitters as em
from sdr_sim import modulation
from sdr_sim.spectrum import VirtualSpectrum

SAMPLE_RATE = 240_000
BAUD = 4800


def peak_offset_hz(samples: np.ndarray, sample_rate: int) -> float:
    """Onde está o pico do espectro, em Hz relativos ao centro."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1.0 / sample_rate))

    return float(freqs[int(np.argmax(spectrum))])


# --- modulação --------------------------------------------------------------


def test_2gfsk_tem_amplitude_constante():
    """FSK é modulação de ângulo: a envoltória não varia. Uma amplitude que
    oscila significa que a fase virou amplitude em algum ponto."""
    bits = np.random.default_rng(1).integers(0, 2, 400)

    wave = modulation.modulate_2gfsk(bits, samples_per_symbol=50)

    assert np.allclose(np.abs(wave), 1.0, atol=1e-5)


def test_2gfsk_respeita_o_indice_de_modulacao():
    """h = 0.5 põe o desvio de pico em um quarto da taxa de símbolo.

    É o número que amarra este simulador ao demodulador da estação: se os dois
    discordarem aqui, os bits saem invertidos ou embaralhados e ninguém vai
    olhar para a modulação primeiro.
    """
    # Todos os bits em 1: desvio constante, máximo, fácil de medir.
    wave = modulation.modulate_2gfsk(np.ones(200, dtype=int), samples_per_symbol=50)

    # Frequência instantânea, em ciclos por amostra.
    delta = np.angle(wave[1:] * np.conj(wave[:-1]))
    measured_hz = float(np.mean(delta[500:-500])) * SAMPLE_RATE / (2 * np.pi)

    assert measured_hz == pytest.approx(BAUD / 4.0, rel=0.05)


def test_preambulo_alternado_sobrevive_ao_filtro():
    """O caso que quase passou batido.

    0x55 alterna a cada bit — é o padrão mais rápido que a modulação precisa
    carregar, e o primeiro a morrer se o pulso gaussiano for largo demais. Com
    um sigma 4x maior que o correto o sinal ainda sai, ainda tem amplitude
    constante, e ainda passa em todo teste de índice de modulação: só o
    preâmbulo desaparece, e o receptor entrega bits enviesados sem syncword.

    Aqui se exige que o desvio de um padrão alternado chegue a uma fração
    razoável do desvio de um padrão constante.
    """
    sps = 50
    alternado = np.tile([0, 1], 100)
    constante = np.ones(200, dtype=int)

    def peak_deviation(bits):
        wave = modulation.modulate_2gfsk(np.asarray(bits), samples_per_symbol=sps)
        delta = np.angle(wave[1:] * np.conj(wave[:-1]))[sps * 10 : -sps * 10]
        return float(np.max(np.abs(delta)))

    assert peak_deviation(alternado) > 0.5 * peak_deviation(constante)


def test_sigma_do_pulso_segue_a_definicao_do_gmsk():
    """sigma = sqrt(ln 2)/(2*pi*BT), em períodos de símbolo."""
    sps, bt = 50, 0.5
    taps = modulation.gaussian_pulse(bt, sps)

    t = (np.arange(len(taps)) - (len(taps) - 1) / 2) / sps
    measured = float(np.sqrt(np.sum(taps * t**2) / taps.sum()))
    expected = np.sqrt(np.log(2.0)) / (2.0 * np.pi * bt)

    assert measured == pytest.approx(expected, rel=0.05)


def test_filtro_gaussiano_tem_area_unitaria():
    taps = modulation.gaussian_pulse(0.5, 50)

    assert taps.sum() == pytest.approx(1.0)


def test_bt_menor_da_pulso_mais_largo_no_tempo():
    """Menos banda ocupada custa mais interferência entre símbolos."""
    estreito = modulation.gaussian_pulse(0.3, 50)
    largo = modulation.gaussian_pulse(0.9, 50)

    # Largura efetiva: quantos taps carregam a maior parte da energia.
    def spread(t):
        return float(np.sum(t > t.max() * 0.1))

    assert spread(estreito) > spread(largo)


def test_fm_tem_amplitude_constante():
    audio = modulation.tone(1000.0, 0.05, SAMPLE_RATE)

    wave = modulation.modulate_fm(audio, SAMPLE_RATE, 50_000.0, preemphasis_us=None)

    assert np.allclose(np.abs(wave), 1.0, atol=1e-5)


# --- enquadramento ----------------------------------------------------------


def test_frame_traz_preambulo_syncword_e_payload():
    frame = em.build_frame(b"\x00\x01", preamble_bytes=4)

    assert len(frame) == (4 + 4 + 2) * 8
    # O syncword começa depois do preâmbulo.
    syncword_bits = em.bytes_to_bits(em.NGHAM_SYNCWORD)
    assert np.array_equal(frame[32:64], syncword_bits)


def test_preambulo_alterna_a_cada_bit():
    """0xAA = 10101010, como o ngham.c define e como aparece numa gravação
    real do FloripaSat-1: os 16 bits antes do primeiro sync são
    1010101010101010."""
    frame = em.build_frame(b"", preamble_bytes=2)

    assert list(frame[:8]) == [1, 0, 1, 0, 1, 0, 1, 0]


def test_bits_saem_msb_primeiro():
    """A ordem em que o NGH_SYNC do ngham.c está escrito."""
    assert list(em.bytes_to_bits(b"\xBA")) == [1, 0, 1, 1, 1, 0, 1, 0]


# --- espectro virtual -------------------------------------------------------


def build_spectrum(center_hz, emitter_hz, **kwargs):
    return VirtualSpectrum(
        center_frequency_hz=center_hz,
        sample_rate_hz=SAMPLE_RATE,
        emitters=[em.carrier(emitter_hz, SAMPLE_RATE)],
        seed=7,
        **kwargs,
    )


def test_portadora_aparece_no_offset_prometido():
    """O teste central: sintonizar 30 kHz abaixo põe o sinal em +30 kHz."""
    spectrum = build_spectrum(145_900_000.0, 145_930_000.0, snr_db=None)

    block = spectrum.block(16384)

    assert peak_offset_hz(block, SAMPLE_RATE) == pytest.approx(30_000.0, abs=100.0)


def test_sintonizar_em_cima_poe_o_sinal_no_centro():
    spectrum = build_spectrum(145_900_000.0, 145_900_000.0, snr_db=None)

    block = spectrum.block(16384)

    assert peak_offset_hz(block, SAMPLE_RATE) == pytest.approx(0.0, abs=100.0)


def test_emissor_fora_da_banda_nao_e_ouvido():
    """Sintonia errada por mais de meia taxa = silêncio, não sinal fraco."""
    spectrum = build_spectrum(145_900_000.0, 146_900_000.0, snr_db=None)

    assert spectrum.visible() == []
    assert np.allclose(np.abs(spectrum.block(4096)), 0.0)


def test_tune_move_o_receptor_e_traz_o_sinal_de_volta():
    """A malha que isto existe para testar: sintonizar certo faz o sinal voltar."""
    spectrum = build_spectrum(145_000_000.0, 145_900_000.0, snr_db=None)
    assert spectrum.visible() == []

    spectrum.tune(145_900_000.0)

    assert len(spectrum.visible()) == 1
    assert spectrum.retune_count == 1
    assert peak_offset_hz(spectrum.block(16384), SAMPLE_RATE) == pytest.approx(0.0, abs=100.0)


def test_tune_invalido_e_recusado():
    spectrum = build_spectrum(145_900_000.0, 145_900_000.0)

    with pytest.raises(ValueError):
        spectrum.tune(0.0)


def test_fase_nao_salta_entre_blocos():
    """Reiniciar a fase a cada bloco criaria um salto que o demodulador leria
    como símbolo errado — o mesmo defeito de borda que a bancada do cano de
    bits expôs do outro lado."""
    spectrum = build_spectrum(145_900_000.0, 145_910_000.0, snr_db=None)

    first = spectrum.block(4096)
    second = spectrum.block(4096)
    joined = np.concatenate((first, second))

    # Numa portadora pura, a diferença de fase entre amostras vizinhas é
    # constante. Um salto na emenda apareceria como um valor fora da linha.
    delta = np.angle(joined[1:] * np.conj(joined[:-1]))
    seam = delta[4090:4100]

    assert np.allclose(seam, delta[100], atol=1e-3)


def test_tempo_corre_pelas_amostras_e_nao_pelo_relogio():
    spectrum = build_spectrum(145_900_000.0, 145_900_000.0)

    spectrum.block(SAMPLE_RATE)

    assert spectrum.elapsed_s == pytest.approx(1.0)


def test_snr_configurado_e_o_snr_medido():
    limpo = build_spectrum(145_900_000.0, 145_900_000.0, snr_db=None).block(32768)
    ruidoso = build_spectrum(145_900_000.0, 145_900_000.0, snr_db=10.0).block(32768)

    signal_power = float(np.mean(np.abs(limpo) ** 2))
    noise_power = float(np.mean(np.abs(ruidoso - limpo) ** 2))
    measured_db = 10.0 * np.log10(signal_power / noise_power)

    assert measured_db == pytest.approx(10.0, abs=1.0)


def test_semente_fixa_produz_a_mesma_corrida():
    """Sem isto, um teste que falha uma vez em dez não é investigável."""
    a = build_spectrum(145_900_000.0, 145_900_000.0, snr_db=10.0).block(2048)
    b = build_spectrum(145_900_000.0, 145_900_000.0, snr_db=10.0).block(2048)

    assert np.array_equal(a, b)


def test_emissor_fora_da_banda_continua_consumindo_a_forma_de_onda():
    """O satélite não para de transmitir porque ninguém está ouvindo — se a
    cadência congelasse, voltar a sintonizar pegaria a rajada do começo."""
    emitter = em.fs2_beacon(146_900_000.0, SAMPLE_RATE, BAUD, b"\x00" * 8, gap_s=0.01)
    spectrum = VirtualSpectrum(145_900_000.0, SAMPLE_RATE, [emitter], snr_db=None)

    spectrum.block(4096)

    assert emitter._cursor == 4096 % len(emitter.waveform)


# --- Doppler ----------------------------------------------------------------


def test_doppler_varre_de_positivo_a_negativo():
    """Positivo enquanto se aproxima, zero na aproximação máxima, negativo
    depois — o sentido importa: invertê-lo faria a correção dobrar o erro."""
    doppler = em.PassDoppler(max_shift_hz=3500.0, pass_duration_s=600.0)

    assert doppler.shift_at(0.0) > 3000.0
    assert doppler.shift_at(300.0) == pytest.approx(0.0, abs=1.0)
    assert doppler.shift_at(600.0) < -3000.0


def test_doppler_move_o_sinal_no_espectro():
    doppler = em.PassDoppler(max_shift_hz=20_000.0, pass_duration_s=10.0)
    emitter = em.carrier(145_900_000.0, SAMPLE_RATE)
    emitter.doppler = doppler
    spectrum = VirtualSpectrum(145_900_000.0, SAMPLE_RATE, [emitter], snr_db=None)

    inicio = peak_offset_hz(spectrum.block(16384), SAMPLE_RATE)
    spectrum.block(SAMPLE_RATE * 9)  # avança quase a passagem inteira
    fim = peak_offset_hz(spectrum.block(16384), SAMPLE_RATE)

    assert inicio > 0
    assert fim < 0


def test_duracao_de_passagem_invalida_e_recusada():
    with pytest.raises(ValueError):
        em.PassDoppler(3500.0, 0.0)


# --- validação de cenário ---------------------------------------------------


def test_taxa_que_nao_divide_o_baud_e_recusada():
    """O resto viraria erro de fase acumulado ao longo da rajada."""
    with pytest.raises(ValueError, match="múltiplo"):
        em.fs2_beacon(145e6, 100_000, 4800, b"\x00")


def test_poucas_amostras_por_simbolo_e_recusado():
    with pytest.raises(ValueError, match="amostras por símbolo"):
        modulation.modulate_2gfsk(np.ones(10, dtype=int), samples_per_symbol=1)
