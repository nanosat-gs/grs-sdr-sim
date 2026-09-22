"""SDR virtual: ocupa o lugar do `grs-iq-rx` sem rádio nenhum.

Substituto direto. Publica o MESMO envelope na mesma porta, então o resto da
estação — demodulador, detector de syncword, gravador — não tem como saber que
do outro lado não há antena:

    PUB :5556   um lote por mensagem, sem frame de tópico, complex64
                little-endian intercalado (`cf32_le`)

E obedece ao `tune`:

    SUB :5557   [b"tune", b"<Hz>"]  do grs-frequency-synthesizer

É isso que fecha a malha de sintonia sem hardware. O Station Manager anuncia
frequência e Doppler na 5581, o sintetizador soma e publica aqui, e o receptor
virtual se move — com o sinal saindo do centro, e sumindo se a sintonia errar
demais. A cadeia inteira passa a ser exercitável numa máquina de mesa.

O que ele NÃO simula, e é bom saber antes de confiar demais: ganho de antena,
figura de ruído, interferência de banda adjacente, multipercurso, e o
assentamento do PLL depois de um retune. Tudo o que ele reproduz é o que a
metade digital da estação consegue observar.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import numpy as np
import zmq

from sdr_sim import emitters as emitters_mod
from sdr_sim.spectrum import VirtualSpectrum

DEFAULT_IQ_BIND = "tcp://*:5556"
DEFAULT_TUNE_SOURCE = "tcp://grs-frequency-synthesizer:5557"

# 240 kS/s, pelos mesmos dois motivos do resto da estação: é válido num RTL-SDR
# de verdade (que só aceita 225001-300000 e 900001-3200000 S/s), e
# 240000/4800 = 50 amostras por símbolo, exato.
DEFAULT_SAMPLE_RATE = 240_000
DEFAULT_BAUD = 4800

# Mesmo valor provisório do resto da estação: 145.9 MHz é a beacon do FS-1,
# usada como exemplo até a coordenação IARU confirmar o FS-2.
DEFAULT_FS2_HZ = 145_900_000.0

DEFAULT_BLOCK_SAMPLES = 8192

_shutdown = threading.Event()


def snr_value(text: str) -> float | None:
    """Aceita um número em dB, ou `none`/`off` para sinal sem ruído.

    Tipo próprio porque `type=float` rejeitaria a palavra que a própria ajuda
    manda usar — uma ajuda que mente é pior do que ajuda nenhuma.
    """
    if text.strip().lower() in {"none", "off", "limpo"}:
        return None

    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"esperava um número em dB ou 'none', veio {text!r}"
        ) from None


def _handle_signal(signum: int, _frame) -> None:
    print(f"[sdr-sim] sinal {signum} recebido, encerrando", flush=True)
    _shutdown.set()


class TuneListener(threading.Thread):
    """Assina o `tune` e move o receptor virtual.

    Em thread própria porque o laço de geração é o caminho de tempo real: um
    `recv` bloqueante no meio dele atrasaria as amostras, e o atraso apareceria
    como falha de demodulação, que é o sintoma mais caro de diagnosticar.
    """

    def __init__(self, address: str, spectrum: VirtualSpectrum) -> None:
        super().__init__(daemon=True, name="tune-listener")
        self._address = address
        self._spectrum = spectrum
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "tune")
        self._socket.setsockopt(zmq.RCVTIMEO, 500)
        self._socket.connect(address)

    def run(self) -> None:
        print(f"[sdr-sim] ouvindo tune em {self._address}", flush=True)

        while not _shutdown.is_set():
            try:
                frames = self._socket.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            if len(frames) != 2:
                continue

            try:
                frequency = float(frames[1].decode())
            except (ValueError, UnicodeDecodeError):
                print(f"[sdr-sim] tune ilegível: {frames[1]!r}", flush=True)
                continue

            try:
                previous = self._spectrum.center_frequency_hz
                self._spectrum.tune(frequency)
            except ValueError as error:
                print(f"[sdr-sim] tune recusado: {error}", flush=True)
                continue

            print(
                f"[sdr-sim] tune {previous / 1e6:.4f} -> {frequency / 1e6:.4f} MHz "
                f"({frequency - previous:+.0f} Hz)",
                flush=True,
            )
            for line in self._spectrum.describe():
                print(line, flush=True)

        self._socket.close()
        self._context.term()


def build_spectrum(args: argparse.Namespace) -> VirtualSpectrum:
    """Monta o cenário: o que está no ar, e onde."""
    doppler = None
    if args.doppler_hz:
        doppler = emitters_mod.PassDoppler(args.doppler_hz, args.pass_duration)

    signals: list = []

    if "fs2" in args.emitters:
        payload = bytes(range(args.payload_bytes))
        signals.append(
            emitters_mod.fs2_beacon(
                frequency_hz=args.fs2_frequency,
                sample_rate_hz=args.sample_rate,
                baud=args.baud,
                payload=payload,
                gap_s=args.burst_gap,
                doppler=doppler,
            )
        )

    if "fm" in args.emitters:
        signals.append(
            emitters_mod.fm_station(
                frequency_hz=args.fm_frequency,
                sample_rate_hz=args.sample_rate,
                amplitude=args.fm_amplitude,
            )
        )

    if "carrier" in args.emitters:
        signals.append(
            emitters_mod.carrier(
                frequency_hz=args.carrier_frequency,
                sample_rate_hz=args.sample_rate,
                amplitude=args.carrier_amplitude,
            )
        )

    if not signals:
        print("[sdr-sim] AVISO: nenhum emissor — só ruído sairá.", flush=True)

    return VirtualSpectrum(
        center_frequency_hz=args.frequency,
        sample_rate_hz=args.sample_rate,
        emitters=signals,
        snr_db=args.snr_db,
        seed=args.seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Os nomes curtos espelham os do grs-iq-rx de propósito: trocar um pelo
    # outro num comando deve ser uma edição de uma palavra.
    parser.add_argument("-f", "--frequency", type=float, default=DEFAULT_FS2_HZ,
                        help="Sintonia inicial do receptor virtual, em Hz")
    parser.add_argument("-s", "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help="Taxa de amostragem, em S/s")
    parser.add_argument("-b", "--block-samples", type=int, default=DEFAULT_BLOCK_SAMPLES,
                        help="Amostras por mensagem ZMQ")

    parser.add_argument("--bind", default=DEFAULT_IQ_BIND,
                        help="Onde publicar o IQ")
    parser.add_argument("--tune-source", default=None,
                        help="PUB do frequency-synthesizer. Omitido = sintonia fixa, "
                             f"sem escutar ninguém. Exemplo: {DEFAULT_TUNE_SOURCE}")

    parser.add_argument("--emitters", default="fs2",
                        help="Lista separada por vírgula: fs2, fm, carrier")
    parser.add_argument("--fs2-frequency", type=float, default=DEFAULT_FS2_HZ)
    parser.add_argument("--fm-frequency", type=float, default=DEFAULT_FS2_HZ + 60_000.0)
    parser.add_argument("--fm-amplitude", type=float, default=0.7)
    parser.add_argument("--carrier-frequency", type=float, default=DEFAULT_FS2_HZ + 20_000.0)
    parser.add_argument("--carrier-amplitude", type=float, default=0.3)

    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--payload-bytes", type=int, default=64)
    parser.add_argument("--burst-gap", type=float, default=0.5,
                        help="Silêncio entre rajadas, em segundos")

    parser.add_argument("--snr-db", type=snr_value, default=20.0,
                        help="Relação sinal-ruído em dB. Use 'none' para sinal limpo.")
    parser.add_argument("--doppler-hz", type=float, default=0.0,
                        help="Desvio Doppler de pico a simular. 0 = sem Doppler.")
    parser.add_argument("--pass-duration", type=float, default=600.0,
                        help="Duração da passagem simulada, em segundos")

    parser.add_argument("--seed", type=int, default=None,
                        help="Semente do ruído. Fixe-a para uma corrida reproduzível.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Segundos a transmitir. 0 = até receber sinal.")

    args = parser.parse_args(argv)
    args.emitters = [name.strip() for name in args.emitters.split(",") if name.strip()]

    unknown = set(args.emitters) - {"fs2", "fm", "carrier"}
    if unknown:
        parser.error(f"emissor desconhecido: {', '.join(sorted(unknown))}")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        spectrum = build_spectrum(args)
    except ValueError as error:
        print(f"[sdr-sim] cenário inválido: {error}", file=sys.stderr, flush=True)
        return 1

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(args.bind)

    print("[sdr-sim] SDR virtual — substituto do grs-iq-rx", flush=True)
    print(f"[sdr-sim] IQ em {args.bind} (cf32_le, {args.block_samples} amostras/msg)", flush=True)
    print(f"[sdr-sim] sintonia {args.frequency / 1e6:.4f} MHz, "
          f"{args.sample_rate / 1000:.0f} kS/s", flush=True)
    print(f"[sdr-sim] SNR {'limpo' if args.snr_db is None else f'{args.snr_db} dB'}"
          f"{f', Doppler +/-{args.doppler_hz:.0f} Hz' if args.doppler_hz else ''}", flush=True)
    print("[sdr-sim] no ar:", flush=True)
    for line in spectrum.describe():
        print(line, flush=True)

    listener = None
    if args.tune_source:
        listener = TuneListener(args.tune_source, spectrum)
        listener.start()
    else:
        print("[sdr-sim] sintonia FIXA (sem --tune-source)", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # PUB descarta o que publica antes de o assinante concluir a conexão. Sem
    # esta espera, as primeiras mensagens somem e alguém gasta a tarde
    # procurando o defeito no demodulador.
    time.sleep(1.0)

    block_duration = args.block_samples / args.sample_rate
    started = time.monotonic()
    blocks = 0

    print(f"[sdr-sim] transmitindo (bloco de {block_duration * 1000:.1f} ms)", flush=True)

    while not _shutdown.is_set():
        publisher.send(spectrum.block(args.block_samples).tobytes())
        blocks += 1

        if args.duration and spectrum.elapsed_s >= args.duration:
            break

        # Ritmo pelo tempo SIMULADO acumulado, e não por um sleep fixo por
        # bloco: com sleep fixo o erro de cada iteração se soma e o fluxo
        # deriva do tempo real ao longo de uma passagem inteira.
        target = started + spectrum.elapsed_s
        slack = target - time.monotonic()
        if slack > 0:
            _shutdown.wait(slack)

    elapsed = time.monotonic() - started
    print(
        f"[sdr-sim] {blocks} blocos, {spectrum.elapsed_s:.1f} s simulados em "
        f"{elapsed:.1f} s reais, {spectrum.retune_count} retunes",
        flush=True,
    )

    publisher.close(linger=1000)
    context.term()

    if listener is not None:
        listener.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
