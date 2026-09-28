# Contexto do projeto

SDR virtual da estação terrestre SpaceLab. **Substituto direto do
`grs-iq-rx`**: mesmo envelope, mesma porta, mesmo `tune`.

## Por que existe

Validar a metade de RF não pode depender de hardware e de uma passagem de LEO:
dura dez minutos, não se repete quando se quer, e não é reproduzível. Aqui o
cenário é declarado, o ruído tem semente, e a mesma corrida sai igual amanhã.

E torna a **sintonia** testável, que é o ganho que não era óbvio: cada emissor
vive numa frequência absoluta e o receptor tem a sua. Sintonize longe e o sinal
sai do centro; muito longe e some.

## Desenho

```
modulation.py   formas de onda, escritas da definição (2GFSK, FM)
emitters.py     o que está no ar: frequência absoluta, forma de onda, Doppler
spectrum.py     soma os emissores na banda-base do receptor; sintonia e ruído
control.py      SimController: estado mutável sob UMA trava (laço, tune, painel)
panel.py        painel web (http.server): GET /api/state, /api/spectrum, POST /api/control
panel.html      a página do painel, servida de dentro do pacote
packets.py      assina :5558 e confere o payload contra o que foi transmitido
main.py         serviço: PUB :5556, SUB :5557 tune, ritmo de tempo real
```

## Decisões, e por quê

**A modulação é independente do demodulador.** Poderia reusar a classe GMSK do
`grs-demodulator` — e a primeira bancada da estação fez isso. Mas um teste em
que o mesmo código modula e demodula prova menos do que parece: dois erros
simétricos se cancelam e o verde é falso. Aqui a modulação vem da definição.

**Frequências absolutas, não offsets.** É o que faz "estar sintonizado"
significar alguma coisa. Um simulador que sempre entrega o sinal centrado não
testa sintonia nenhuma.

**Emissor fora da banda não é ouvido, em vez de dobrar para dentro dela.** Um
receptor de verdade faz aliasing. Simular isso seria mais fiel e produziria um
sinal fantasma no lugar errado — e alguém gastaria uma tarde caçando um bug que
o simulador inventou. Silêncio é a mentira menos perigosa.

**O tempo corre pelas amostras, não pelo relógio.** Se a máquina engasgar, o
sinal não deve pular: o Doppler acompanha as amostras que de fato saíram.

**O `tune` roda em thread separada.** Um `recv` bloqueante no laço de geração
atrasaria as amostras, e o atraso apareceria como falha de demodulação — o
sintoma mais caro de diagnosticar que existe.

**Semente fixa por padrão.** Sem ela, um teste que falha uma vez em dez não é
investigável.

## Armadilhas conhecidas

- **`grs-sdr-sim` e `grs-iq-rx` não sobem juntos.** Os dois BINDAM a :5556.
  Por isso profiles diferentes: `rxsim` e `rx`.
- **O sigma do pulso gaussiano é `sqrt(ln 2)/(2·pi·BT)`, em períodos de
  símbolo.** Errar esse número por um fator de 4 produz um sinal que ainda tem
  amplitude constante, ainda passa no teste de índice de modulação, e ainda sai
  do ar — só que o preâmbulo alternado é apagado pelo filtro e o receptor
  entrega bits enviesados sem syncword. Foi exatamente o que aconteceu aqui. O
  teste `test_preambulo_alternado_sobrevive_ao_filtro` existe por causa disso.
- **PUB descarta o que publica sem assinante conectado.** Por isso o serviço
  espera um segundo antes de transmitir. Sem isso as primeiras mensagens somem
  e alguém procura o defeito no demodulador.
- **Toda mudança em execução passa pela trava do `SimController`.** O laço de
  geração a segura durante `spectrum.block()`; tune e painel também. Sem ela,
  um retune no meio do bloco dá metade das amostras numa sintonia e metade
  noutra — o demodulador lê símbolo errado e ninguém reproduz depois.
- **`SimController.apply` é tudo-ou-nada.** Valida o pedido inteiro antes de
  mudar qualquer coisa: um campo inválido não pode deixar o simulador meio
  alterado enquanto o painel sugere que nada mudou.
- **Não nomeie atributo de subclasse de `Thread` como `_started`.** O
  `threading.Thread` usa esse nome para um Event interno; sobrescrevê-lo
  quebra o `.start()`. Aconteceu no `PacketMonitor` e só a execução no compose
  pegou — daí o teste que inicia a thread de verdade.
- **O que o painel mediu, e é problema do demodulador, não do simulador:**
  mesmo sem ruído, ~8% dos pacotes chegam com payload divergente e ~7% das
  rajadas nem são detectadas; com 20 dB, ~25% divergem. É a deriva de
  sincronismo de tempo registrada em `docs/rx-datapath.md` do `grs-station`.
- **O que ele NÃO simula:** ganho de antena, figura de ruído, interferência de
  banda adjacente, multipercurso, e o assentamento do PLL depois de um retune.
  O Doppler é um MODELO (curva em S), não propagação orbital — quem calcula
  Doppler de verdade é a `spacelab-tracking`, no Station Manager.

## Convenções

- Comentários e mensagens de commit em português; código em inglês.
- Testes conferem **propriedades do sinal** (offset, índice de modulação, SNR
  medido, continuidade de fase), não que o processo roda. Um simulador que
  mente contamina todo teste feito com ele.
