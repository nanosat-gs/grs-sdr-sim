# GRS SDR Sim

SDR virtual da estação terrestre SpaceLab. **Substituto direto do
`grs-iq-rx`**: publica o mesmo envelope na mesma porta e obedece ao mesmo
`tune`, então o resto da estação não distingue um do outro.

```
                         SUB :5557 "tune"
                                │
                                ▼
  cenário declarado ──▶ [ grs-sdr-sim ] ── PUB :5556 (cf32_le) ──▶ demodulador ──▶ ...
  (emissores, ruído,
   Doppler)
```

## Por que existe

Validar a metade de RF não pode depender de hardware e de uma passagem de LEO:
a passagem dura dez minutos, não se repete quando você quer, e não é
reproduzível. Aqui o cenário é declarado, o ruído tem semente, e a mesma
corrida sai igual amanhã.

E há um ganho que só aparece depois: **a sintonia passa a ser testável.** Cada
emissor vive numa frequência absoluta e o receptor tem a sua — sintonize longe
e o sinal sai do centro, sintonize muito longe e ele some. Com o
`--tune-source` apontado para o `grs-frequency-synthesizer`, a cadeia inteira
(Station Manager → sintetizador → receptor) roda numa máquina de mesa.

## Usando

```bash
# FS-2 sintético, sintonia fixa
python -m sdr_sim.main

# passagem com Doppler, ruído e três emissores no ar
python -m sdr_sim.main --emitters fs2,fm,carrier --doppler-hz 3500 \
    --pass-duration 600 --snr-db 15

# obedecendo ao sintetizador
python -m sdr_sim.main --tune-source tcp://grs-frequency-synthesizer:5557

# o FS-2 vira a ISS: Doppler da órbita real, calado abaixo do horizonte
python -m sdr_sim.main --orbit-norad 25544

# a próxima passagem da ISS, começando agora
python -m sdr_sim.main --orbit-norad 25544 --orbit-mode next-pass

# o oscilador do satélite errado em +1200 Hz: o que o ajuste fino da estação tem de achar
python -m sdr_sim.main --orbit-norad 25544 --fs2-offset-hz 1200

# um General Telemetry (0x10) do FS-2 em quadro NGHam de verdade, a 4800 baud
python -m sdr_sim.main --frame general-telemetry --baud 4800
```

No compose: `SIM_ORBIT_NORAD=25544` (e `SIM_ORBIT_MODE=next-pass`, se quiser)
antes do `docker compose --profile rxsim up`. As coordenadas da estação vêm
das mesmas `GS_*` do TC Scheduler e do Station Manager.

### Um simulador por rádio

A estação tem duas cadeias de recepção, e o compose sobe um simulador para
cada uma:

| Serviço | Rádio | Portadora / baud | Painel | Erro de oscilador |
|---|---|---|---|---|
| `grs-sdr-sim` | VHF (beacon) | 145,9 MHz / 1200 | <http://localhost:8090> | `SIM_FS2_OFFSET_HZ` |
| `grs-sdr-sim-uhf` | UHF (dados) | 468,4 MHz / 4800 | <http://localhost:8093> | `SIM_UHF_FS2_OFFSET_HZ` |

Cada um segue o `tune` do sintetizador do seu rádio e, com
`--station-tuning-channel vhf|uhf`, compara o próprio Doppler com o do canal
dele na :5581 (`doppler.vhf`, `doppler.uhf`).

### O que o FS-2 transmite: `--frame`

| Modo | Depois do syncword | Para quê |
|---|---|---|
| `pattern` (padrão) | `00 01 02 ...` embaralhado com a sequência CCSDS. **Não é NGHam**: sem size tag nem Reed-Solomon | O DSP: demodulador e detector |
| `general-telemetry` | Um General Telemetry (`0x10`) do FS-2 em **quadro NGHam de verdade**, montado com o PyNGHam (`fs2_frames.py`) | O caminho inteiro, até a telemetria decodificada pelo `grs-telemetry-decoder` |

No compose, o simulador VHF usa `pattern` e o UHF, `general-telemetry`
(`SIM_UHF_FRAME`).

No `general-telemetry`, cada rajada leva um pacote novo: o relógio do satélite
é o da máquina, a página do log de dados conta as rajadas, e temperaturas,
bateria e painéis variam numa senoide de 5 min. Com `--frame-timestamp` o
pacote volta a ser fixo, para uma corrida reproduzível. O layout do `0x10` é
reescrito aqui, sem importar o decoder: se os dois divergirem, o teste ponta a
ponta do `grs-station` acusa.

O `pattern` é embaralhado com a mesma tabela `ccsds_poly` do firmware do TTC
2.0 (ver `emitters.ccsds_scramble`). Sem embaralhar, o payload tinha 37,5% de
uns e enviesava em −50 Hz a medida do bloco FFT — o satélite real embaralha,
então o simulador também.

## Painel de controle

Com `--panel-port 8090`, o simulador serve um painel web (no compose:
<http://localhost:8090>, publicado só em 127.0.0.1). Tudo muda com o
simulador rodando, sem reiniciar:

- **espectro e cascata ao vivo** do que sai na :5556, com a posição esperada
  de cada emissor marcada;
- **sintonia** do receptor virtual (valor exato ou passos de ±1/±10 kHz);
- **SNR**, ou sinal sem ruído;
- **emissores** ligados/desligados e amplitude — os três existem sempre;
  `--emitters` só diz quais começam ligados;
- **Doppler** do FS-2, de dois jeitos:
  - **órbita real** — NORAD (TLE do CelesTrak) ou TLE colado; "onde ele está
    agora" ou "próxima passagem, começando agora"; calado abaixo do horizonte
    (desmarcável). Mostra elevação, distância, velocidade radial e o desvio;
  - **modelo** — a curva em S, com pico e duração escolhidos;
- **comparação com a estação**: com `--station-tuning-source
  tcp://station-manager:5581` (ligado no compose, com
  `--station-tuning-channel` do rádio), o painel põe o Doppler que o Station
  Manager anuncia ao lado do que o simulador impõe. Só compara, nunca aplica;
- **erro do oscilador do FS-2**: a portadora sai deslocada da nominal, e o
  Doppler é calculado sobre ela (`--fs2-offset-hz` na partida);
- **transmissão do FS-2**: contínua (uma rajada a cada ~0,67 s) ou
  **manual** — calado até você pedir "enviar 1 pacote" (ou N). Também na
  partida: `--fs2-mode manual`.

Com `--packets-source tcp://grs-syncword-detector:5558`, o painel também
assina a saída do detector e **confere cada pacote contra o payload que o
simulador transmitiu** — os 64 bytes, não só o começo — e mostra enviados,
corretos, divergentes e os que não chegaram. É a malha fechada: baixe o SNR,
ou envie um pacote só, e veja no mesmo lugar o que aconteceu com ele.

O painel usa só a biblioteca padrão (`http.server`); as rotas são
`GET /api/state`, `GET /api/spectrum` e `POST /api/control`. Não tem
autenticação: é ferramenta de bancada.

Emissores disponíveis:

| Nome | O que é | Para quê |
|---|---|---|
| `fs2` | rajadas 2GFSK com enquadramento NGHam | exercitar demodulador e detector |
| `fm` | portadora FM com uma melodia | teste de ouvido do caminho de áudio |
| `carrier` | portadora pura, sem modulação | conferir sintonia direto no espectro |

## Modo bancada: o modem do rádio da bancada RF

Na etapa 1 da bancada RF do FS-2, o rádio da estação é um Raspberry Pi com um
Si4463, e o satélite é uma Bluepill com outro Si4463. O Si4463 transmite e
recebe bytes, não IQ. No modo bancada, este simulador roda **no Pi** e faz a
conversão, com o mesmo código que ele já usa para fingir o satélite:

```
grs-uplink ─:5559 IQ─▶ demodula ─▶ quadro ─▶ serviço do rádio ─SPI─ Si4463 ))) Bluepill
cadeia     ◀─:5556 IQ── modula  ◀─ quadro ◀─ serviço do rádio ◀─────────────────┘
```

- A `:5559` mantém o contrato. O `ack` só sai depois de o rádio confirmar a
  rajada, com os horários **do rádio**. Sem confirmação, `timeout` ou `error`.
- O `satellite.py` não responde e o FS-2 do simulador não transmite nada por
  conta própria: o que vai ao ar na cadeia é só o que o rádio recebeu.
- O painel ganha o cartão "Rádio da bancada": estado do rádio, canal em
  escuta, TCs mandados e quadros recebidos com o RSSI.
- O quadro recebido é remodulado com **4 bytes de preâmbulo**, como o Si4463
  do FS-2 transmite, e não com os 32 de folga do simulador
  (`--preamble-bytes` muda). Com 32, o sincronismo de tempo da estação tinha
  256 símbolos para travar e a bancada escondia uma perda de quadros que o
  satélite de verdade teria.

```bash
# no Pi: os dois simuladores ao lado do serviço do rádio (API em 5600/5601)
python -m sdr_sim.main --bind tcp://*:5556 -f 145900000 --fs2-frequency 145900000 --baud 1200 \
  --tx-bind tcp://*:5559 --bench-radio tcp://127.0.0.1:5600 --bench-rx tcp://127.0.0.1:5601 \
  --responses-channel vhf --tune-source tcp://<PC>:5575 --panel-port 8090
python -m sdr_sim.main --bind tcp://*:5566 -f 468400000 --fs2-frequency 468400000 --baud 4800 \
  --bench-rx tcp://127.0.0.1:5601 --responses-channel uhf --tune-source tcp://<PC>:5576 --panel-port 8093

# sem RF e sem o firmware da Bluepill: o rádio falso responde com o satellite.py
python -m sdr_sim.bench_radio_stub --keys-file keys.test.json --beacon-period 30
```

**A API do serviço do rádio** (o contrato completo está em `bench.py`):

| | |
|---|---|
| REQ/REP `:5600` | `{"op": "tx", "channel", "frame_hex"}` → `{ok, started_at, ended_at}` ou `{ok: false, error}`; `{"op": "health"}` |
| PUB `:5601` | `[b"vhf"\|b"uhf", quadro, {"rssi", "received_at"}]` |

O quadro é o size tag + codeword do NGHam: o que o Si4463 escreve depois do
syncword. O `bench_radio_stub` aceita também `{"op": "sat", "fail_next": -2}`,
`{"op": "sat", "drop_next": true}` e `{"op": "beacon"}`.

**O gateway** (`python -m sdr_sim.gateway`) roda na estação, com o nome e o IP
fixo do rádio (`grs-iq-rx`, `grs-iq-rx-uhf`). Ele repassa o IQ do Pi à cadeia
e a `:5559` ao Pi, e republica o `tune` do sintetizador numa porta do host
para o Pi assinar. Sem IQ do Pi há mais de 3 s, a `:5559` responde `error`
na hora, com o motivo.

## A modulação é independente de propósito

Este repositório **não** usa o DSP do `grs-demodulator`. Poderia — a classe
GMSK de lá modula e demodula. Mas um teste em que o mesmo código modula e
demodula prova menos do que parece: dois erros simétricos se cancelam e o teste
passa em verde com o sinal errado.

Aqui a modulação é escrita a partir da definição. Se o demodulador recuperar
estes bits, é porque os dois concordam sobre o que é 2GFSK.

## O que ele NÃO simula

Vale saber antes de confiar demais: ganho de antena, figura de ruído,
interferência de banda adjacente, multipercurso, e o assentamento do PLL depois
de um retune. E um emissor fora da banda simplesmente não é ouvido, em vez de
dobrar para dentro dela como faria um receptor de verdade — silêncio é a
mentira menos perigosa.

O Doppler da órbita real é calculado aqui, com geometria própria (ver
`orbit.py`), e não pela `spacelab-tracking` que o Station Manager usa para
corrigi-lo: a mesma conta dos dois lados cancelaria os próprios erros. A
intensidade do sinal não varia com a distância (perda de percurso não é
simulada): o satélite no horizonte chega tão forte quanto no zênite.

## Testes

```bash
pip install -e .[dev]
pytest -q
```

Um simulador é instrumento de medida: se ele mentir, todo teste feito com ele
mente junto. Por isso a suíte confere propriedades do sinal — o offset da
portadora, o índice de modulação, o SNR medido, a continuidade de fase entre
blocos — e não só que o processo roda.
