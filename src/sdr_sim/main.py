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

Com `--orbit-norad` (ou pelo painel), o FS-2 simulado passa a ser um satélite
de verdade: o Doppler vem da órbita real, calculado aqui com geometria
própria, e o sinal some abaixo do horizonte. Com `--station-tuning-source`,
o painel põe ao lado o Doppler que o Station Manager anuncia — só para
comparar, nunca para aplicar (ver `station_tuning.py`).

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
from sdr_sim.control import SimController
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

    def __init__(self, address: str, controller: SimController) -> None:
        super().__init__(daemon=True, name="tune-listener")
        self._address = address
        self._controller = controller
        self._spectrum = controller.spectrum
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
                # Mesma trava do laço de geração e do painel: um retune nunca
                # cai no meio de um bloco.
                with self._controller.lock:
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


def optional_int(text: str) -> int | None:
    """Inteiro, ou vazio para "não usar". O compose passa `--orbit-norad=` com
    a variável vazia quando ninguém pediu órbita."""
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"esperava um inteiro, veio {text!r}") from None


def expected_payload(args: argparse.Namespace) -> bytes:
    """O que o FS-2 sintético põe depois do syncword. O monitor de pacotes
    confere a saída do detector contra exatamente isto."""
    if args.frame == "general-telemetry":
        # Quadro NGHam de verdade (fs2_frames.py). Já sai embaralhado pelo
        # próprio NGHam — passar pelo ccsds_scramble aqui embaralharia duas
        # vezes, e o decodificador receberia lixo com CRC errado.
        from sdr_sim import fs2_frames

        timestamp = args.frame_timestamp if args.frame_timestamp is not None else int(time.time())
        return fs2_frames.ngham_frame(fs2_frames.general_telemetry(timestamp))

    # Embaralhado como o NGHam embaralha (ver emitters.ccsds_scramble): bits
    # equilibrados, como os do satélite.
    return emitters_mod.ccsds_scramble(bytes(range(args.payload_bytes)))


def live_telemetry(args: argparse.Namespace) -> bool:
    """General Telemetry com o relógio andando: um pacote novo por rajada.
    Com --frame-timestamp, o pacote é um só, fixo — para corrida reproduzível."""
    return args.frame == "general-telemetry" and args.frame_timestamp is None


def packet_check(args: argparse.Namespace):
    """O que o monitor de pacotes confere. Pacote fixo: os bytes exatos. Pacote
    que muda a cada rajada: que seja um General Telemetry NGHam válido."""
    if live_telemetry(args):
        from sdr_sim import fs2_frames

        return fs2_frames.is_general_telemetry_frame
    return expected_payload(args)


def build_spectrum(args: argparse.Namespace) -> VirtualSpectrum:
    """Monta o cenário: o que está no ar, e onde.

    Os três emissores existem SEMPRE; `--emitters` só decide quais começam
    ligados. É o que deixa o painel ligar um emissor que não estava no
    comando de partida sem reiniciar o simulador.
    """
    doppler = None
    if args.doppler_hz:
        doppler = emitters_mod.PassDoppler(args.doppler_hz, args.pass_duration)

    payload_source = None
    if live_telemetry(args):
        from sdr_sim import fs2_frames

        payload_source = fs2_frames.GeneralTelemetryStream()

    signals = [
        emitters_mod.fs2_beacon(
            frequency_hz=args.fs2_frequency,
            sample_rate_hz=args.sample_rate,
            baud=args.baud,
            payload=expected_payload(args),
            gap_s=args.burst_gap,
            doppler=doppler,
            payload_source=payload_source,
        ),
        emitters_mod.fm_station(
            frequency_hz=args.fm_frequency,
            sample_rate_hz=args.sample_rate,
            amplitude=args.fm_amplitude,
        ),
        emitters_mod.carrier(
            frequency_hz=args.carrier_frequency,
            sample_rate_hz=args.sample_rate,
            amplitude=args.carrier_amplitude,
        ),
    ]

    for emitter in signals:
        emitter.enabled = emitter.name in args.emitters
    signals[0].continuous = args.fs2_mode == "continuous"
    signals[0].carrier_offset_hz = args.fs2_offset_hz

    if not args.emitters:
        print("[sdr-sim] AVISO: nenhum emissor ligado — só ruído sairá.", flush=True)

    return VirtualSpectrum(
        center_frequency_hz=args.frequency,
        sample_rate_hz=args.sample_rate,
        emitters=signals,
        snr_db=args.snr_db,
        seed=args.seed,
    )


def start_uplink(args: argparse.Namespace, controller: SimController):
    """Liga o satélite simulado: o TX da :5559 e/ou o caminho das respostas.

    :return: o `respond(radio, quadro)` em uso, para o monitor de pacotes saber
    que respostas vão passar pelo detector.
    """
    from sdr_sim import uplink
    from sdr_sim.satellite import SatelliteModel, load_keys

    fs2 = controller.emitter("fs2")
    publisher = uplink.ResponsePublisher(args.responses_bind) if args.responses_bind else None

    def respond(radio: str, frame: bytes) -> None:
        if radio == args.responses_channel:
            fs2.enqueue(frame)          # o próprio rádio: direto na fila
        elif publisher is not None:
            publisher.publish(radio, frame)
        else:
            print(f"[sdr-sim] resposta pela {radio.upper()} descartada: sem --responses-bind",
                  flush=True)

    if args.tx_bind:
        keys = load_keys(args.sat_keys_file)
        station = uplink.UplinkStation(
            SatelliteModel(keys), controller,
            nominal_hz=args.uplink_frequency or args.fs2_frequency, baud=args.uplink_baud,
            respond=respond, shutdown=_shutdown,
            tolerance_hz=args.uplink_tolerance_hz or uplink.DEFAULT_TOLERANCE_HZ,
            rx_max_bytes=args.tc_rx_max_bytes)
        controller.uplink = station
        uplink.TxServer(args.tx_bind, station, _shutdown).start()
        print(f"[sdr-sim] satélite simulado ouvindo uplink em {station.nominal_hz / 1e6:.4f} MHz, "
              f"{args.uplink_baud} baud, {len(keys)} chave(s)", flush=True)

    if args.responses_source:
        uplink.ResponseListener(args.responses_source, args.responses_channel, fs2.enqueue,
                                _shutdown).start()

    return respond


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
    parser.add_argument("--fs2-offset-hz", type=float, default=0.0,
                        help="Erro do oscilador do FS-2: a portadora sai deslocada disto da "
                             "nominal (o que o ajuste fino da estação tem de achar)")
    parser.add_argument("--fm-frequency", type=float, default=DEFAULT_FS2_HZ + 60_000.0)
    parser.add_argument("--fm-amplitude", type=float, default=0.7)
    parser.add_argument("--carrier-frequency", type=float, default=DEFAULT_FS2_HZ + 20_000.0)
    parser.add_argument("--carrier-amplitude", type=float, default=0.3)

    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--payload-bytes", type=int, default=64,
                        help="Tamanho do padrão do --frame pattern")
    parser.add_argument("--frame", choices=["pattern", "general-telemetry"], default="pattern",
                        help="pattern: 00 01 02 ... embaralhado, para o DSP (não é NGHam). "
                             "general-telemetry: um General Telemetry (0x10) do FS-2 em "
                             "quadro NGHam de verdade, que o grs-telemetry-decoder decodifica.")
    parser.add_argument("--frame-timestamp", type=int, default=None,
                        help="Fixa o General Telemetry num instante (Unix epoch): o mesmo "
                             "pacote em toda rajada, para corrida reproduzível. Omitido = um "
                             "pacote novo por rajada, com o relógio da máquina e valores "
                             "variando.")
    parser.add_argument("--burst-gap", type=float, default=0.5,
                        help="Silêncio entre rajadas, em segundos")
    parser.add_argument("--fs2-mode", choices=["continuous", "manual"], default="continuous",
                        help="continuous: rajadas sem parar. manual: silêncio até o painel "
                             "pedir um pacote (POST /api/control {\"send_packets\": 1}).")

    parser.add_argument("--snr-db", type=snr_value, default=20.0,
                        help="Relação sinal-ruído em dB. Use 'none' para sinal limpo.")
    parser.add_argument("--doppler-hz", type=float, default=0.0,
                        help="Desvio Doppler de pico a simular. 0 = sem Doppler.")
    parser.add_argument("--pass-duration", type=float, default=600.0,
                        help="Duração da passagem simulada, em segundos")

    parser.add_argument("--orbit-norad", type=optional_int, default=None,
                        help="NORAD ID do satélite que o FS-2 simulado imita. O TLE vem do "
                             "CelesTrak e o Doppler, da órbita real vista da estação (GS_*). "
                             "Substitui --doppler-hz. Vazio = sem órbita.")
    parser.add_argument("--orbit-mode", choices=["realtime", "next-pass"], default="realtime",
                        help="realtime: o satélite onde ele está agora. next-pass: adianta o "
                             "relógio do satélite para a próxima passagem começar agora.")
    parser.add_argument("--orbit-ignore-horizon", action="store_true",
                        help="Ouvir o satélite mesmo abaixo do horizonte — para testar a malha "
                             "de Doppler com a passagem sintética do station_demo.py.")
    parser.add_argument("--station-tuning-channel", default="",
                        help="Rádio que este simulador imita (vhf, uhf...): compara com "
                             "freq.<canal>/doppler.<canal>. Vazio = os tópicos sem canal.")
    parser.add_argument("--station-tuning-source", default=None,
                        help="PUB de sintonia do Station Manager (:5581). O painel compara o "
                             "Doppler anunciado lá com o do simulador. Omitido = sem comparação.")

    parser.add_argument("--seed", type=int, default=None,
                        help="Semente do ruído. Fixe-a para uma corrida reproduzível.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Segundos a transmitir. 0 = até receber sinal.")

    parser.add_argument("--panel-port", type=int, default=0,
                        help="Porta do painel web de controle. 0 = sem painel.")
    parser.add_argument("--panel-host", default="0.0.0.0",
                        help="Interface do painel. Dentro do container, 0.0.0.0; quem "
                             "restringe a 127.0.0.1 é o mapeamento de porta do compose.")
    parser.add_argument("--packets-source", default=None,
                        help="PUB de raw packets do detector (:5558). O painel confere o "
                             "que chega lá contra o que foi transmitido. Omitido = sem "
                             "conferência.")

    # --- uplink: o satélite simulado ouvindo a estação ------------------------
    parser.add_argument("--tx-bind", default=None,
                        help="REP do TX (contrato da :5559, o mesmo do grs-iq-rx/usrp): o "
                             "grs-uplink manda a rajada e o satélite simulado a recebe. "
                             "Omitido = sem uplink. Exemplo: tcp://*:5559")
    parser.add_argument("--uplink-frequency", type=float, default=None,
                        help="Frequência em que o rádio de bordo ouve, em Hz. Padrão: "
                             "--fs2-frequency (o FS-2 ouve no TTC_1, a mesma da VHF)")
    parser.add_argument("--uplink-baud", type=int, default=1200)
    parser.add_argument("--uplink-tolerance-hz", type=float, default=None,
                        help="Residual de frequência que o rádio de bordo ainda ouve. Um "
                             "modelo, não uma medida do Si446x. Padrão 1500 Hz.")
    parser.add_argument("--tc-rx-max-bytes", type=int, default=0,
                        help="Bytes que o rádio de bordo lê depois do syncword (o 80 do "
                             "uplink_manager.c v1.0, TODO-1). 0 = sem limite.")
    parser.add_argument("--sat-keys-file", default=None,
                        help="Chaves HMAC do satélite simulado (o mesmo JSON do grs-uplink). "
                             "Sem arquivo, todo TC privado é rejeitado.")
    parser.add_argument("--responses-bind", default=None,
                        help="PUB das respostas do satélite, tópico = rádio (vhf/uhf). Só no "
                             "simulador que recebe o uplink. Exemplo: tcp://*:5567")
    parser.add_argument("--responses-source", default=None,
                        help="SUB das respostas: as do rádio --responses-channel entram na "
                             "próxima rajada deste simulador. Exemplo: tcp://grs-sdr-sim:5567")
    parser.add_argument("--responses-channel", default=None,
                        help="Rádio que este simulador transmite (vhf/uhf). Padrão: "
                             "--station-tuning-channel, ou vhf.")

    args = parser.parse_args(argv)
    args.emitters = [name.strip() for name in args.emitters.split(",") if name.strip()]
    args.responses_channel = (args.responses_channel or args.station_tuning_channel or "vhf").lower()

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

    controller = SimController(spectrum, args.block_samples)
    controller.tune_source = args.tune_source

    listener = None
    if args.tune_source:
        listener = TuneListener(args.tune_source, controller)
        listener.start()
    else:
        print("[sdr-sim] sintonia FIXA (sem --tune-source)", flush=True)

    if args.orbit_norad:
        mode = args.orbit_mode.replace("-", "_")
        try:
            controller.apply({"orbit": {
                "norad_id": args.orbit_norad, "mode": mode,
                "horizon": not args.orbit_ignore_horizon,
            }})
            orbital = controller.emitter("fs2").doppler
            look = orbital.look_at(spectrum.elapsed_s)
            print(f"[sdr-sim] FS-2 imita {orbital.satellite.name} (NORAD {args.orbit_norad}), "
                  f"{mode}: el {look.elevation_deg:.1f}°, Doppler "
                  f"{look.doppler_hz(args.fs2_frequency):+.0f} Hz", flush=True)
        except ValueError as error:
            # Sem rede não é motivo para não subir: o painel tenta de novo.
            print(f"[sdr-sim] AVISO: órbita não carregada ({error}); FS-2 sem Doppler.",
                  flush=True)

    if args.station_tuning_source:
        from sdr_sim.station_tuning import StationTuningMonitor

        controller.station_tuning = StationTuningMonitor(
            args.station_tuning_source, _shutdown, channel=args.station_tuning_channel)
        controller.station_tuning.start()

    respond = None
    if args.tx_bind or args.responses_source:
        respond = start_uplink(args, controller)

    if args.packets_source:
        # Import tardio: sem --packets-source, nada disto é carregado.
        from sdr_sim.packets import PacketMonitor

        check = packet_check(args)
        if respond is not None:
            # As respostas do satélite passam pelo mesmo detector: não são divergência.
            from sdr_sim.uplink import is_response_frame

            base = check
            check = (lambda payload: (base(payload) if callable(base)
                                      else payload[: len(base)] == base) or is_response_frame(payload))
        controller.packets = PacketMonitor(args.packets_source, check, _shutdown)
        controller.packets.start()

    panel = None
    if args.panel_port:
        from sdr_sim.panel import start_panel

        panel = start_panel(controller, args.panel_host, args.panel_port)
        print(f"[sdr-sim] painel em http://{args.panel_host}:{args.panel_port}/", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # PUB descarta o que publica antes de o assinante concluir a conexão. Sem
    # esta espera, as primeiras mensagens somem e alguém gasta a tarde
    # procurando o defeito no demodulador.
    time.sleep(1.0)

    block_duration = args.block_samples / args.sample_rate
    started = time.monotonic()

    print(f"[sdr-sim] transmitindo (bloco de {block_duration * 1000:.1f} ms)", flush=True)

    while not _shutdown.is_set():
        publisher.send(controller.next_block().tobytes())

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
        f"[sdr-sim] {controller.blocks} blocos, {spectrum.elapsed_s:.1f} s simulados em "
        f"{elapsed:.1f} s reais, {spectrum.retune_count} retunes",
        flush=True,
    )

    if panel is not None:
        panel.shutdown()

    publisher.close(linger=1000)
    context.term()

    if listener is not None:
        listener.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
