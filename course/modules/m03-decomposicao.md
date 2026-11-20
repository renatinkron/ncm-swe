# M03 — Diagnóstico por decomposição

**Objetivo**: aprender a quebrar um pipeline em componentes medíveis
separadamente — porque "o modelo não funciona" nunca é um diagnóstico,
é um sintoma.

## O pipeline como cadeia de suspeitos

```
texto ──BPE──▶ tokens ──encoder VQ──▶ códigos ──VQ decode──▶ tokens ─▶ texto
                    │                                    ▲
                    └────────── GERADOR prevê códigos ────┘
```

Três componentes podem falhar independentemente:

1. **Encoder/canal VQ** — códigos preservam a informação?
2. **Decoder** — consegue reconstruir tokens a partir de códigos certos?
3. **Gerador** — consegue prever códigos certos?

Se você mede só a saída final ("parseia?"), um fracasso pode ser qualquer
um dos três — e consertar o errado queima semanas.

## As medições que isolaram cada um

**Canal (encoder→decoder com códigos verdadeiros):**
```
decoder antigo:   recovery 65%    → decoder era o gargalo
pós-repair/boost: recovery 96.8%  → resolvido
prova lossless:   overfit em 64 held-out = 100% → canal íntegro
```

**Canal-only fim-a-fim (códigos verdadeiros → decode → ast.parse):**
```
antes do boost: 25/127 (19.7%)
depois:         71/127 (55.9%)   → canal sustenta Python real
```

**Gerador (teacher-forced, acurácia por nível):**
```
nas lições:   l1-l7 88-96%, l8-l11 67-83% → fino = fraqueza
no corpus:    held-out fine 10.2%, l0 = 0% → gargalo residual é ele
```

Conclusão decomposta: canal provado OK → sobra o gerador. Cada medição
atacou UM componente com os outros congelados ou fornecidos como ouro.

## A técnica central: teacher-forcing + ouro

Para medir o componente X sem contaminar com os erros dos outros:

- Dê ao componente a **entrada verdadeira** (códigos gold para o decoder,
  prefixo verdadeiro para o gerador) em vez da saída do componente anterior.
- Meça a saída contra o alvo verdadeiro.

O erro do pipeline em cadeia = composição dos erros; o erro de cada
componente isolado = o que você precisa consertar.

## Caso real — quando a decomposição revelou dado errado

Na medição de 2026-11-16, a acurácia por nível foi medida primeiro nos
dados de LIÇÃO (gen treinado nelas): 23%. Alarmante — até notar que o
modelo agora treina no corpus de CÓDIGO. Medir o checkpoint novo nos dados
velhos mistura esquecimento com progresso. A medição honesta exigiu:

```
baseline gen_lessons2 em held-out de CÓDIGO:  4.3%
gen_code2 em held-out de CÓDIGO:             10.2%  ← progresso real 2.4x
gen_code2 em rows de TREINO:                 18.3%  ← gap = generalização
```

Três números, três perguntas distintas respondidas.

## Lab 3 — Decompor antes de propor conserto

**Objetivo**: mapear a cadeia de suspeitos de um problema aberto.

O modelo atual produz samples que ainda não compilam. Liste:

1. Cada componente que pode estar errando (encoder? codebook? gerador?
   decoder? detokenizer? o próprio `ast.parse`?)
2. Para cada um: qual medição isola sua contribuição?
3. Quais já foram medidas (ver handoff) e o que provaram?
4. Qual componente ainda não tem medição limpa?

```bash
# O harness de decomposição por nível existe e é parametrizável:
sed -n 27,30p scripts/_gen_acc.py   # GEN_CK, CODES, OFFSET por env
```

**Decida**: qual é o próximo número que falta medir — e por quê?

---

*Anterior: [M02](m02-instrumentacao.md) · Próximo: [M04 — Métricas honestas](m04-metricas.md)*
