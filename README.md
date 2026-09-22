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
```

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

O Doppler é um MODELO (curva em S), não propagação orbital. Quem calcula
Doppler de verdade é a `spacelab-tracking`, no Station Manager.

## Testes

```bash
pip install -e .[dev]
pytest -q
```

Um simulador é instrumento de medida: se ele mentir, todo teste feito com ele
mente junto. Por isso a suíte confere propriedades do sinal — o offset da
portadora, o índice de modulação, o SNR medido, a continuidade de fase entre
blocos — e não só que o processo roda.
