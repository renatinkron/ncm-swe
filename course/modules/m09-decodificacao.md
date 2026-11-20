# M09 — Decodificação como multiplicador

**Objetivo**: entender quando a resposta para "o modelo erra" não é
"treine mais" — é mude o jogo na saída: amostre N vezes, verifique
determinísticamente, repare o melhor.

## A virada estratégica que as medições forçaram

A medição de 2026-11-16 fechou uma porta e abriu outra:

```
Fechada: held-out fine acc 10.2%, l0 = 0%, deficiência difusa
         → exigir 641/641 códigos exatos é a estratégia errada
Aberta:  o erro é ALEATÓRIO por amostra (superfície), não sistemático
         → cada sample erra em posições diferentes → combinar tentativas
```

Se o erro fosse sistemático, N amostras errariam juntas. Sendo aleatório,
a diversidade das falhas vira recurso.

## Best-of-N + verificador determinístico

Para código, o verificador é **grátis e exato** — não precisa de reward
model nem de opinião:

```python
def score(sample):
    try:    ast.parse(sample); compile(sample)
    except: return fracasso_deterministico
    return sucesso  # ou: contagem de erros → "menos quebrado"
```

A matemática da colheita: se cada amostra tem probabilidade `p` de parse,
com N amostras independentes:

```
P(pelo menos 1 parseia) = 1 - (1-p)^N
p=5%:  N=16 → 56%  |  p=20%: N=16 → 97%
```

Mesmo quando nenhuma compila, devolver a **menos quebrada** já vale.
Custo medido neste hardware: ~2-5s/sample → N=16 serial ~1min,
N=8 em paralelo ~15s. Não pesa o modelo — é wrapper de inferência.

## O laço generate→verify→repair

```
1. GERADOR: amostra N candidatos
2. VERIFICADOR: ast.parse descarta/reclassifica (determinístico, ms)
3. RANKING: melhor candidato (parseia? menos erros?)
4. REPAIR HEAD (opcional): 1 passada corrigindo erros COMUNS do canal
   — treinado nos pares (corrompido → correto) que o próprio modelo gera
5. LOOP: se falhou, reamostra ou repara de novo (2-3 voltas)
```

O repair head é a peça elegante: os dados de treino dele são **infinitos e
grátis** — gere N saídas do modelo atual, classifique o que quebrou, e o
par (quebrado → alvo real do dataset) vira exemplo supervisionado. O
modelo aprende com o próprio erro — na forma densa e barata, não em RL
esparsa e lenta (ver M01 para por que RL clássico seria pior aqui).

## O que a decomposição garante antes disso

Best-of-N só funciona porque o canal foi provado lossless (M03) — se o
decoder perdesse informação, nem o sample perfeito de códigos geraria
Python. **A estratégia de decodificação pressupõe componentes abaixo
provados.** É por isso que a ordem do projeto foi: canal → gerador →
decodificação.

## Lab 9 — Estimar seu N

**Objetivo**: dimensionar best-of-N com seu `p` medido, não otimista.

Dos dados do projeto: canal-only sustenta 55.9% de parse com códigos
gold; gerador ~10% exact-match held-out. Estime `p` por amostra para o
pipeline composto (dica: é produto de probabilidades — e o canal já
cobra 1.7% de corrupção tolerável, não zero).

Calcule N para `P(acerto) ≥ 90%` com seu `p`, depois compare com o custo:
`N × 3s/sample ÷ paralelismo`.

**Observe**: como `p` pequeno explode N — e por que subir `p` (mais
épocas, repair) multiplica mais que subir N.

**Decida**: em que ponto a conta diz "pare de amostrar, volte a treinar"?

---

*Anterior: [M08](m08-handoff.md) · Próximo: [M10 — Capstone](m10-capstone.md)*
