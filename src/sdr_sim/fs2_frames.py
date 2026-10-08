"""Quadros do FS-2 de verdade: pacote do OBDH2 -> NGHam, como o satélite manda.

O modo padrão do simulador (`--frame pattern`) transmite `00 01 02 ...`
embaralhado depois do syncword: serve ao DSP, mas não é NGHam — não tem size
tag nem Reed-Solomon, e nenhum decodificador o aceitaria. Este módulo monta o
quadro inteiro (`--frame general-telemetry`), para o caminho de recepção
poder ser exercitado até a telemetria decodificada.

O NGHam é o do PyNGHam (spacelab-ufsc / mgm8, LGPLv3), o mesmo que o
`grs-telemetry-decoder` usa para decodificar. Codificar e decodificar com a
mesma biblioteca não prova a biblioteca — quem prova é a fixture REAL do
GOLDS-UFSC, nos testes do decoder. Aqui o objetivo é outro: provar o cano.

O layout do General Telemetry é o do `obdh2/firmware/app/tasks/general_telemetry.c`
(111 bytes, big-endian), escrito de novo aqui de propósito, sem importar o
decoder: se os dois divergirem, o teste ponta a ponta acusa.
"""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Callable

import pyngham

GENERAL_TELEMETRY_ID = 0x10
CALLSIGN = b" PY0EFS"   # 7 bytes, espaço à esquerda: CONFIG_SATELLITE_CALLSIGN

# ID, callsign e os 103 bytes de dados, campo a campo (spec §4).
GENERAL_TELEMETRY_FORMAT = (
    ">B7s"          # ID, callsign
    "IHHHBHB"       # OBDH: timestamp, temp, corrente, tensão, causa reset, contador, último TC
    "HH"            # TTC: temp do rádio, RSSI
    "HH"            # antena: temp, status
    "HHBH"          # EPS: temp µC, corrente, causa reset, contador
    "HHH"           # tensão dos painéis
    "HHHHHH"        # corrente dos painéis
    "BBB"           # MPPT
    "HHhhHH"        # barramento, bateria: tensão, corrente, média, acumulada, temp do monitor
    "BB"            # aquecedores
    "BBB"           # EDC principal, payload principal, secundário
    "II"            # páginas: log, SBCD
    "II"            # leitura de sensores, instante da posição
    "hhh"           # latitude, longitude, altitude
    "III"           # última TLE, último contato, próximo TC agendado
    "BB"            # fila de TC, modo
)

# Preâmbulo (4 x 0xAA) + syncword que o PyNGHam põe na frente. O emissor põe
# o preâmbulo e o syncword dele (emitters.build_frame), então estes saem.
NGHAM_PREFIX = 8


# Período das grandezas que variam (temperaturas, bateria, painéis). Curto de
# propósito: num pacote por rajada e no painel atualizando a cada 15 s, um
# período de órbita de verdade (~95 min) pareceria constante na tela.
VARIATION_PERIOD_S = 300


def general_telemetry(timestamp: int, sequence: int = 0) -> bytes:
    """Um General Telemetry plausível no instante `timestamp` do satélite.

    Os valores são função do instante e de `sequence` (quantos pacotes já
    saíram), e nada mais: o mesmo par dá sempre os mesmos bytes, e pacotes
    diferentes dão bytes diferentes — que é o que permite ver, na ponta do
    cano, que cada pacote foi decodificado de verdade e não é uma cópia.

    O que varia, e como:

    - relógio do satélite e os instantes derivados dele: andam com `timestamp`;
    - página do log de dados: 1000 + `sequence` — um contador visível;
    - temperaturas, tensão e corrente da bateria, painéis e MPPT: senoides de
      período VARIATION_PERIOD_S. A corrente da bateria cruza o zero (carga e
      descarga), o que exercita o int16;
    - o resto (posição, modo, contadores de reset) fica fixo.

    Num instante múltiplo do período, a fase é zero e os valores são os de
    referência (temperatura do OBDH 298 K, bateria 7600 mV e -250 mA...).
    """
    phase = math.sin(2 * math.pi * (timestamp % VARIATION_PERIOD_S) / VARIATION_PERIOD_S)
    sun = max(0.0, phase)          # painéis só geram com o "sol" do lado positivo

    def wave(base: float, amplitude: float) -> int:
        return int(round(base + amplitude * phase))

    return struct.pack(
        GENERAL_TELEMETRY_FORMAT,
        GENERAL_TELEMETRY_ID, CALLSIGN,
        timestamp, wave(298, 6), wave(45, 5), 3300, 0, 3, 0x40,
        wave(300, 8), 0,
        wave(290, 15), 0x0000,
        wave(299, 6), wave(120, 20), 0, 5,
        wave(4500, 300), wave(4510, 300), wave(4520, 300),
        *(int(round(base + 400 * sun)) for base in (100, 110, 120, 130, 140, 150)),
        wave(80, 15), wave(81, 15), wave(82, 15),
        5000, wave(7600, 150), wave(-250, 400), wave(-200, 300), 1500, wave(293, 5),
        0, 0,
        1, 1, 0,
        1000 + sequence, 500,
        timestamp - 5, timestamp - 30,
        -27, -48, 520,
        timestamp - 86_400, timestamp - 3_600, 0,
        0, 1,
    )


class GeneralTelemetryStream:
    """Um General Telemetry NOVO a cada chamada: o satélite transmitindo.

    É o que o emissor chama no início de cada rajada. O relógio do satélite é
    o da máquina (`clock`), e a sequência conta as rajadas.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.sequence = 0

    def __call__(self) -> bytes:
        frame = ngham_frame(general_telemetry(int(self._clock()), self.sequence))
        self.sequence += 1
        return frame


def is_general_telemetry_frame(data: bytes) -> bool:
    """O que o monitor de pacotes confere quando o conteúdo muda a cada
    rajada: os bytes depois do syncword são um quadro NGHam válido (CRC ok)
    com um General Telemetry do FS-2 dentro."""
    payload, _, _ = pyngham.PyNGHam().decode(list(data))
    return len(payload) == 111 and bytes(payload[:8]) == bytes([GENERAL_TELEMETRY_ID]) + CALLSIGN


def ngham_frame(packet: bytes) -> bytes:
    """Size tag + codeword embaralhado: o que vem depois do syncword no ar."""
    encoded = pyngham.PyNGHam().encode(list(packet))
    if not encoded:
        raise ValueError(f"PyNGHam recusou um pacote de {len(packet)} bytes (máx. 220)")
    return bytes(encoded[NGHAM_PREFIX:])
