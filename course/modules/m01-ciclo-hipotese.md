# M01 — O ciclo hipótese→evidência

**Objetivo**: estruturar experimentos que *decidem* algo — cada ciclo deve
terminar com uma hipótese confirmada, refutada ou refinada, nunca com
"interessante, vamos ver".

## O ciclo em cinco passos

```
1. HIPÓTESE falsificável   — "o canal VQ perde informação irrecuperável"
2. MÉTRICA que decide      — recovery de tokens em rows held-out
3. BASELINE honesta        — recovery antes da intervenção: 65%
4. ISOLAR a variável       — mexer no decoder, congelar tudo mais
5. EVIDÊNCIA → decisão     — 96.8% → hipótese REFINADA: perda era do
                              decoder (generalização), não do codebook
```

O passo que amadores pulam é o **5**: anotar a decisão e o que ela descarta.
Um experimento sem decisão registrada é só tempo de CPU gasto.

## Caso real 1 — o teste de overfit como faca diagnóstica

Registrado em `SESSION_HANDOFF.md` (RUN em curso 2026-11-15):

- Hipótese anterior: "os códigos VQ têm teto intrínseco de perda"
- Teste barato e decisivo: **overfit intencional** do decoder em 64 rows
  held-out. Se os códigos perdessem informação, nem overfit salvaria.
- Resultado: **100% de recuperação em 200 steps** → os códigos são
  LOSSLESS. O platô era generalização do decoder (problema de dados),
  não perda do canal.

Uma tarde de overfit em 64 linhas valeu mais que semanas de "aumentar o
modelo". A pergunta certa era pequena.

## Caso real 2 — refutar antes de escalar

Registrado no handoff (kdistill): a hipótese "selecionar exemplos difíceis
por loss melhora a destilação" foi testada contra **baseline aleatória**:

```
seleção aleatória:  CE 6.63
seleção por loss:   CE 6.75   ← pior
base (sem sel.):    CE 7.93
```

Ambas ajudaram vs a base — mas a seleção "inteligente" *perdeu* para a
aleatória. "Difícil ≠ transferível". Refutada na escala testada, a hipótese
foi arquivada como resultado negativo — e economizou o trabalho de montar
uma pipeline de curadoria que não teria pago.

## A regra de ouro

> Nunca rode um experimento sem poder dizer: *"se o resultado for X, eu
> faço A; se for Y, eu faço B".* Se ambos os resultados levam à mesma
> ação, o experimento era desnecessário.

## Lab 1 — Desenhar antes de rodar

**Objetivo**: praticar o ciclo completo no papel, antes de gastar CPU.

Pegue a questão aberta atual do projeto (ver `SESSION_HANDOFF.md`,
"PRÓXIMA AÇÃO"): *o gerador chegará a código compilável por esta via?*

Escreva, no formato dos cinco passos:

- Qual hipótese falsificável está em jogo?
- Qual métrica decide (dica: o projeto usa `ast.parse` como oráculo —
  por que CE sozinho não serve?)
- Qual é a baseline (0/16, 0/32 parseable registrados)?
- Que variável isolar (gerador? canal? decodificação?)
- Que resultado levaria a qual decisão?

**Decida**: seu desenho tem uma ação diferente para cada outcome possível?
Se não, redesenhe.

---

*Anterior: [M00](m00-caminho-probatorio.md) · Próximo: [M02 — Instrumentação antes de otimização](m02-instrumentacao.md)*
