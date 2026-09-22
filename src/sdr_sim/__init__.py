"""SDR virtual da estação terrestre SpaceLab.

Substituto direto do `grs-iq-rx`: publica o mesmo envelope na mesma porta e
obedece ao mesmo `tune`, então o resto da estação não distingue um do outro.

Existe porque validar a metade de RF não pode depender de hardware e de uma
passagem de LEO: a passagem dura dez minutos, não se repete quando se quer, e
não é reproduzível. Aqui o cenário é declarado, o ruído tem semente, e a mesma
corrida sai igual amanhã.
"""
