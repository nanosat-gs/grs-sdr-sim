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
```

No compose: `SIM_ORBIT_NORAD=25544` (e `SIM_ORBIT_MODE=next-pass`, se quiser)
antes do `docker compose --profile rxsim up`. As coordenadas da estação vêm
das mesmas `GS_*` do TC Scheduler e do Station Manager.

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
  tcp://station-manager:5581` (ligado no compose), o painel põe o Doppler que o
  Station Manager anuncia ao lado do que o simulador impõe. Só compara, nunca
  aplica;
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
