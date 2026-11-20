# M07 — Resultados negativos como ativos

**Objetivo**: tratar hipóteses refutadas como capital de pesquisa —
elas são o que impede a equipe (ou você, daqui a um mês) de pagar o
mesmo custo duas vezes.

## O preço de esquecer um negativo

Se um resultado negativo não é registrado, ele não existiu — e a próxima
pessoa razoável vai tentar de novo, porque a hipótese *era* razoável.
Negativo registrado com contexto = experimento que nunca mais precisa
ser repetido.

## Acervo de negativos deste projeto

### kdistill — "difícil ≠ transferível"

Hipótese: selecionar exemplos difíceis (alto loss) para destilar melhora
o aprendizado vs seleção aleatória.

```
base sem seleção:   CE 7.93
seleção aleatória:  CE 6.63   ← melhor
seleção por loss:   CE 6.75   ← pior que aleatória!
```

Refutada **na escala testada** — repare na qualificação: o registro não
diz "nunca funciona", diz "não funcionou aqui, neste regime". Isso deixa
a porta aberta sem deixar a ilusão viva.

### probe l0 — features de LLM congelado não transferem

RUN-035: hipótese de que ativações de um modelo maior ajudariam a prever
o nível de conceito (l0). Resultado: **NEGATIVO** — registrado com seção
própria no handoff, metodologia e tudo. Economizou a linha inteira de
pesquisa "transplante de features".

### torch.compile — ~0% em GEMM-bound

Não foi bug (depois de contornar o stride assert do Inductor no VQ):
foi medida limpa — 15.3 vs 15.5s/step. O negativo confirmou a predição
do roofline e encerrou a frente "otimizar a engine". Ver M05.

### "teto intrínseco do canal" — hipótese refutada por overfit

O platô de ~93% de recovery parecia perda do codebook. O teste de overfit
(100% em held-out) provou que era generalização de dados. A hipótese
antiga estava **escrita no handoff** — foi corrigida, não apagada.

## Como registrar um negativo que presta

Mínimo vital por entrada:

```
1. Hipótese exata que foi testada (não uma caricatura)
2. Baseline contra a qual perdeu (número)
3. Resultado obtido (número)
4. Escopo da refutação ("na escala testada", "neste dataset")
5. O que ela descarta / o que ela NÃO descarta
```

O item 5 é o mais importante: "seleção por loss não ajuda" não implica
"curriculum não ajuda" — implica especificamente que loss-como-proxy-de-
dificuldade não transferiu. Precisão no escopo evita tanto repetir o
erro quanto desistir da direção certa.

## Lab 7 — Escrever seu próprio negativo

**Objetivo**: praticar o registro de refutação.

```bash
# O formato vivo do projeto:
grep -n -A8 "NEGATIVO\|refutad" SESSION_HANDOFF.md | head -30
```

Pegue uma hipótese sua recente que falhou (qualquer projeto). Escreva a
entrada nos 5 itens acima — com especial cuidado no item 5.

**Observe**: quantas vezes o handoff repete números de baseline ao lado
do resultado — nunca "piorou", sempre "6.75 vs 6.63".

**Decida**: seu registro permite a alguém decidir *sem repetir o
experimento* se a hipótese vale outra tentativa em escala diferente?

---

*Anterior: [M06](m06-recursos.md) · Próximo: [M08 — Handoff e continuidade](m08-handoff.md)*
