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
orbit.py        Doppler da órbita real: TLE, estação, geometria própria, passagens
station_tuning.py  escuta o Doppler anunciado na :5581 — só para comparar
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

**O Doppler da órbita é calculado aqui, não pela spacelab-tracking.** Ela é
quem CORRIGE o Doppler no Station Manager; se também o IMPUSESSE, um erro nela
(sinal trocado, rotação da Terra esquecida) apareceria dos dois lados e o
espectro ficaria centrado com a estação errada. A velocidade radial sai de
diferença finita entre duas distâncias, sem velocidade nem ω×r — o caminho que
a spacelab-tracking não usa. Só o SGP4 é compartilhado: é ele que define o que
um TLE significa. Medido ao vivo: as duas contas concordam em 1–5 Hz (o resto
é o tick de 1 s da estação).

**O simulador não aplica o Doppler que a estação anuncia.** Seria circular: o
desvio imposto seria exatamente o corrigido, e o erro sairia zero sempre. A
:5581 é assinada só para o painel comparar (`station_tuning.py`), e só no modo
tempo real — na "próxima passagem" o relógio do satélite está adiantado horas.

**Um simulador por rádio.** A estação tem uma cadeia por faixa; no compose
há o `grs-sdr-sim` (VHF, beacon a 1200 baud) e o `grs-sdr-sim-uhf` (dados a
4800 baud), os dois imitando o mesmo satélite. `--station-tuning-channel`
escolhe com que canal da :5581 cada um se compara (`doppler.vhf`, ...); o
prefixo do ZMQ faria `doppler` receber os dois, por isso o tópico é comparado
inteiro.

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
- **Os símbolos do simulador começam alinhados na amostra 0; os de um
  satélite, não.** Um sincronismo de tempo que não rastreia nada passa em
  qualquer teste feito só com este simulador. Foi assim que o ganho errado
  do M&M do `grs-demodulator` sobreviveu; a `tools/bancada_demod.py` do
  `grs-station` atrasa o sinal (`--offset`) por causa disso.
- **No modo manual, a rajada em curso termina antes de calar.** Cortá-la ao
  meio transmitiria um pacote truncado, contado como enviado e perdido sem
  culpa do cano. Por isso o painel espera ~2 s para zerar a contagem depois
  de trocar para manual.
- **O SGP4 não recusa TLE velho.** Propaga um TLE de 2008 até hoje e devolve
  uma posição inventada, sem erro. Por isso `Satellite.check_age` recusa TLE a
  mais de 30 dias do instante simulado. Foi um teste que mostrou isso.
- **Buscar TLE e achar a próxima passagem ficam FORA da trava.** É rede e uma
  varredura de 48 h; segurar a trava congelaria o laço e abriria um buraco no
  IQ. `_orbit_action` monta tudo antes e só troca o Doppler sob a trava.
- **A portadora cadastrada no TC Scheduler precisa ser a do FS-2 simulado.** A
  estação sintoniza em (portadora cadastrada + Doppler); se ela for 145,8 MHz e
  o FS-2 estiver em 145,9, sobra 100 kHz que nenhuma correção de Doppler tira.
  O painel avisa.
- **O que ele NÃO simula:** ganho de antena, figura de ruído, interferência de
  banda adjacente, multipercurso, perda de percurso (o sinal não enfraquece no
  horizonte), e o assentamento do PLL depois de um retune.

## Convenções

- Comentários e mensagens de commit em português; código em inglês.
- Testes conferem **propriedades do sinal** (offset, índice de modulação, SNR
  medido, continuidade de fase), não que o processo roda. Um simulador que
  mente contamina todo teste feito com ele.
