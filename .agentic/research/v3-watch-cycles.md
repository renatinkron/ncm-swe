# v3 — ciclos de monitoramento 2h (RUN-025)

Cadência: a cada ~2h (ou a cada ckpt novo — evals caem ~2,5h): colher ponto,
pesquisa direcionada à pergunta que os dados levantam, decidir
keep/adjust/kill. Gatilhos de decisão no plano RUN-025.

Baseline: v2 dup 0.1095, gap +1.2357, gt 4.4124.

| ciclo | hora | último ponto | dup | gap | decisão |
|---|---|---|---|---|---|
| 1 | 10 nov 16:45 | @5999 | 0.0542 | +1.089 | **KEEP** — ver análise |
| 2 | 10 nov ~18:45 | @6499 | 0.0505 | +1.101 | **ADJUST** — v4 GLAT-lite lançado |
| 3 | 10 nov ~23:30 | v3 @6999 | 0.0644 | +1.105 | **KEEP** — v4 em voo; v5-intra candidato |
| 4 | 11 nov ~00:15 | v4 @6998 | 0.0578 | **+1.616** | **KEEP 1 ponto** — negativo no pareado |
| 5 | 11 nov ~02:15 | v3 @7499 | 0.0515 | +1.183 | **KEEP** — aguarda v4@7498 p/ veredito |
| 6 | 11 nov ~04:15 | — | — | — | **KILL v4** — veredito antecipado |
| 7 | 11 nov ~06:00 | v3 @7999 | **0.046** | **+0.987** | **KEEP** — gap rompeu 1,0! |
| 8 | 11 nov ~08:45 | v3 @8499 | 0.0526 | +1.167 | **KEEP** — sub-1.0 era variância |
| 9 | 11 nov ~11:00 | v3 @8999 | 0.0528 | +1.109 | **KEEP** — 1º ponto do critério de parada |
| 10 | 11 nov ~13:45 | v3 @9499 | 0.0576 | +1.090 | **STOP** — critério de parada disparou |

## Ciclo 10 (final) — 11 nov ~13:45 UTC

**v3 @9499**: dup 0.0576, distinct 0.7603, self 5.306, gt **4.216**
(melhor da série), gap +1.090 — 2º ponto consecutivo na faixa
0.95–1.20 → **critério de parada antecipada disparado: v3 STOPPED.**

Container parado e removido; watcher morto. Terminal =
`artifacts/v3_eval/ckpt_step9499.pt` (snapshot + eval completos).
Economia: ~3.500 steps ≈ ~24h de CPU — converge estável há 4.500.

## Veredito da série v3 (10 pontos, n=8 cada)

| métrica | v2 | v3 final (@9499) | Δ |
|---|---|---|---|
| dup_rate | 0.1095 | 0.0576 | **−47%** |
| distinct | 0.663 | 0.760 | +15% |
| self CE | 5.648 | 5.306 | −6% |
| gt CE limpo | 4.412 | **4.216** | −4.4% (melhor que baseline) |
| exposure gap | +1.236 | +1.090 → platô ~1.10 | −12% (saturado @~4k) |

Conclusões para o paper:
1. Corrupção linear oficial (0.75→0.25) em warm-start REDUZ gagueira
   ~metade e gap ~12%, mantendo/melhorando CE limpo — melhoria real
   mas saturada em +1.1 nats neste orçamento.
2. Corrupção com predições-próprias (v4 GLAT-lite) é PIOR (+46% gap
   no pareado) — a FORMA do ruído importa; converge com o negativo
   de neighbour-corruption do WPS. Predições "corretas demais" não
   ensinam recuperação.
3. Refinement no decode (Mask-Predict-style) é a alavanca restante —
   exige `intra=True` no treino → candidato v5.
4. Arco narrativo global permanece ausente — separado do problema
   de robustez local (l0/escala), como previsto.

## Ciclo 9 — 11 nov ~11:00 UTC

**v3 @8999**: dup 0.0528, distinct 0.6808, self 5.384, gt **4.2752**,
gap +1.109 — dentro da faixa pré-registrada (0.95–1.20), 1º dos 2
pontos exigidos para a parada antecipada. gt CE empatado no melhor
valor (4.275) — converge para ~4.27.

Se @9499 também cair na faixa → STOP: declara-se convergência em
gap ~1.1 / dup ~0.05 / gt ~4.28 e economiza ~5h até o fim nominal.

## Ciclo 8 — 11 nov ~08:45 UTC

**v3 @8499**: dup 0.0526, distinct 0.7487, self 5.443,
gt **4.2757** (melhor da série), gap +1.167 — rebotou acima de 1,0.

Quadro real: gap oscila **0.99–1.18, média ~1.10** desde @4999
(convenção adotada adiante: "flat" = faixa inclusive 0.95–1.20 →
@4999 em diante conta) — o sub-1.0 do @7999 foi variância n=8, não
rompimento. Piso do schedule
linear neste orçamento ≈ +1.1 nats; dup_rate piso ≈ 0.05.
gt CE segue caindo monotonicamente (modelo aprende, robustez saturou).

**Pré-registro de parada antecipada**: se @8999 e @9499 permanecerem
em 0.95–1.20, a trajetória é conclusiva — parar v3 em ~@9499,
declarar "gap estabilizou ~1.1, dup ~0.05" e economizar ~10h de CPU.
A história do paper já está completa: corrupção linear oficial
reduziu stutter ~50% e gap ~15%, sem resolver multimodalidade residual;
corrupção com predições-próprias (v4) é pior que uniforme.

## Ciclo 7 — 11 nov ~06:00 UTC

**v3 @7999**: dup **0.046** (mínimo da série), distinct 0.7512,
self 5.3135, gt 4.3265, gap **+0.987** — primeiro ponto abaixo de 1,0.

Releitura do "platô": 5 pontos em 1.05–1.18 podem ter sido descida
LENTA, não saturação — @7999 confirma que o gap continua cedendo.
Valida a decisão de manter o v3 como braço principal até o fim.
(O timing é ~1,5k steps após a morte do v4 — coincidência, não
causalidade: v3 nunca viu o v4.)

Veredito v4 permanece válido: a comparação pareada foi feita no MESMO
orçamento total (@6998 vs @6999) — o +1,616 do v4 não pode ser
explicado por "esperar mais", só por transient de adaptação, que o
próprio próximo ponto do v4 teria testado (braço resumível se preciso).

Próximo: @8499 (~2h) — se gap seguir <1,0, a tese "corrupção linear
oficial reduz exposure bias continuamente" se fortalece; se voltar
>1,05, era variância n=8.

## Ciclo 6 — 11 nov ~04:15 UTC

**Decisão: KILL v4 (container parado/removido, watcher morto).**

Racional: +46% de degradação do gap no pareado @6998 é efeito grande
(não ruído n=8), converge com o negativo independente do WPS
(neighbour-corruption), e a confirmação @7498 custaria ~5h atrasando o
v3 ~2× por contenção. O braço é RESUMÍVEL — snapshot
`artifacts/v4_eval/ckpt_step6998.pt` preservado; se a hipótese
"transient de 500 steps" importar depois, retomamos do snapshot.

Aprendizado para o paper: corromper o condicionamento com códigos
amostrados do próprio modelo (GLAT-lite adaptado a hierarquia) PIORA o
exposure gap vs controle uniforme — segundo datapoint independente de
que a FORMA do ruído importa mais que sua presença. Mecanismo
plausível: predições do modelo são "corretas demais" → condicionamento
permanece próximo do manifold limpo → não ensina recuperação; e/ou o
modelo aprende a superestimar a confiabilidade do condicionamento,
amplificando erros em cascata na inferência.

v3 agora com CPU livre (~17-25s/step) → @7999 em ~2,5h, fim (~13000)
em ~36h. Braço principal segue até o final para a trajetória completa.

## Ciclo 5 — 11 nov ~02:15 UTC

**v3 @7499**: dup 0.0515, distinct 0.6807, self 5.489, gt **4.306**,
gap +1.183 — sexto ponto flat (1.05–1.18). Platô consolidado; gt ainda
melhora lentamente (4.55→4.31 no braço v3).

O controle pareado para v4@7498 já existe (este ponto): quando v4@7498
cair (~4h), veredito kill/keep conforme gap > ou < ~1.4.

**Decisão: KEEP** — sem ação até v4@7498.

## Ciclo 4 — 11 nov ~00:15 UTC

**Comparação pareada v4@6998 vs v3@6999** (mesmo orçamento desde o
warm-start; único delta = fonte da corrupção):

| métrica | v3 uniforme | v4 self-corruption |
|---|---|---|
| dup_rate | 0.0644 | 0.0578 |
| distinct | 0.7156 | 0.6723 |
| self CE | 5.423 | 5.868 |
| gt CE | 4.318 | 4.252 |
| **gap** | **+1.105** | **+1.616** |

**Resultado negativo**: GLAT-lite tornou o modelo MAIS sensível aos
próprios erros (+1,62 vs +1,11), não mais robusto. gt CE marginalmente
melhor (4,25) — trade-off errado. Converge com WPS "neighbour
corruption" (piorou GenPPL 61–69 vs 43–49): corromper condicionamento
com códigos plausíveis/não-uniformes pode AMPLIFICAR o mismatch —
o modelo aprende a confiar demais no condicionamento.

Hipóteses para o mecanismo: (a) predições do modelo são "corretas
demais" — não ensinam recuperação de erro real; (b) transient de 500
steps após 6,5k de uniforme; (c) loss 5,0 vs 5,5 não comparável
(corruption distribution mais fácil).

Bug corrigido no ciclo: watch_v4.sh FIRST=6999→6998 (primeiro boundary
do offset 6499 é 6998); eval @6998 rodado manualmente.

**Decisão: KEEP v4 mais 1 ponto** (@7498 ~5h) — se gap seguir >1,4,
KILL v4 e documentar negativo. v3 segue como braço principal.
Candidato seguinte: v5 intra (masked conditioning + Mask-Predict no
decode) — a única alavanca ainda não testada para o resíduo ~1,1.

## Ciclo 3 — 10 nov ~23:30 UTC

**v3 @6999** (controle): dup 0.0644 (oscilando no ruído n=8 ~0,05–0,065),
distinct 0.7156, self 5.4227, gt **4.3181** (sempre melhorando),
gap **+1.105** — platô firme: 5 pontos em 1.05–1.11.

Leitura: o MODELO segue melhorando (gt ↓ constante); o que saturou é o
benefício da corrupção uniforme sobre robustez/exposure. Reforça v4.

**v4 em voo**: ~55s/step (2 forwards), step ~6770 → primeiro eval
@6998 em ~4h.

**Pergunta do ciclo**: se o resíduo é multimodalidade intra-nível,
existe correção no lado do DECODE sem retreino?

**Fontes**:
- Mask-Predict / CMLM (Ghazvininejad et al. 2019, arXiv 1904.09324):
  re-predizer iterativamente as posições de MENOR confiança — +4 BLEU
  vs NAR puro, ~1 BLEU do AR. Mas exige modelo treinado com masked
  conditioning (CMLM).
- ReMDM (arXiv 2503.00307) / T2M (arXiv 2604.18738) / Posterior
  Refinement (FMLM+): mesma família — posições erradas caem
  consistentemente no conjunto de baixa confiança; remask-and-repredict
  é inference-time compute scaling.
- Estudo empírico ACL 2025 (aclanthology 2025.acl-long.1443): caveat —
  confiança predita correlaciona imperfeitamente com qualidade.

**Limitação arquitetural identificada**: refinement dentro de um nível
exige que o modelo condicione posições do MESMO nível entre si — nosso
modelo é intra=False (independência condicional); re-amostrar o mesmo
nível com o mesmo condicionamento não muda nada. O mecanismo `intra`/
`mask_emb` EXISTE no generator (projetado RUN-016) mas nunca treinado.

**Candidato v5**: treinar com `intra=True` → destrava Mask-Predict-style
refinement por nível (decided/masked codes via _intra_add). A fila fica:
v4 (self-corruption, em voo) → v5 (intra + refinement no decode).
Decisão de v5 após o primeiro ponto do v4 (@6998).

**Decisão: KEEP** — aguardar v4@6998 + v3@7499.

## Ciclo 2 — 10 nov ~18:45 UTC

**@6499**: dup **0.0505** (novo mínimo), distinct 0.7178, self CE 5.4726,
gt CE **4.3713** (< v2 baseline 4.412 — modelo agora teacher-forced
MELHOR que v2 e com metade do stutter), gap = **+1.101**.

**Platô confirmado**: gap +1.064 → +1.048 → +1.089 → +1.101 — 4 pontos
flat na faixa 1.05–1.11. A corrupção uniforme linear deu o que tinha.
Gatilho pré-registrado disparado.

**Decisão: ADJUST** — lançado braço **v4 GLAT-lite** em paralelo:
- `--corrupt-schedule self` (commit ae086af): corrompe condicionamento
  com códigos amostrados das PRÓPRIAS predições do modelo (não ruído
  uniforme) — GLAT de Qian et al. 2021 adaptado a hierarquia de códigos
- warm-start do snapshot @6499, offset 6499, 3000 steps, THREADS=6
  (v3 segue com ~10 → v3 continua como braço de controle uniforme)
- primeiro eval v4 em local 499 → label ~6998 (convenção −1 vs v3)
- v3 labels 6999/7499/... servem como controle pareado no mesmo
  horizonte de steps totais
- watcher v4 (pid 1860710) + eval_ckpt_v4.sh ativos; watcher antigo
  duplicado do v3 (pid 1382545) morto neste ciclo

**Comparação pareada planejada**: v4@6998 vs v3@6999 — mesmo orçamento
total de steps desde v2, único delta = fonte dos códigos de corrupção
(uniforme vs amostrados-do-modelo). Se v4 reduzir gap/dup abaixo do
controle → evidência própria de que erro-do-modelo > ruído uniforme
para exposure bias hierárquico.

**Custo**: v4 ~2× por step (forward extra no corrupt) × threads
menores → ~55-70s/step, ~500 steps ≈ 8-10h até o primeiro ponto.

## Ciclo 1 — 10 nov ~16:45 UTC

**Estado da série**
| ckpt | dup_rate | gap | gt CE |
|---|---|---|---|
| v2 | 0.1095 | +1.236 | 4.412 |
| @4499 | 0.0792 | +1.246 | 4.546 |
| @4999 | 0.0587 | +1.064 | 4.491 |
| @5499 | 0.0596 | +1.048 | 4.436 |
| @5999 | 0.0542 | +1.089 | 4.405 |

- dup halved & no piso ~0,055; gap melhorou ~0,19 e agora oscila 1,05–1,09
  (platô emergente?); clean gt CE totalmente recuperado (4,405 < v2 4,412!)

**Pergunta levantada**: se o gap estagna ~1,0, existe técnica mais forte que
corrupção uniforme implementável no nosso orçamento?

**Fontes**:
- GLAT (Qian et al., ACL 2021 — arXiv 2008.07905): glancing sampling —
  corrompe condicionamento com as PRÓPRIAS predições amostradas do modelo
  (2ª passada de decode em treino), não com ruído uniforme. NAT→0,9 BLEU
  do AT, 7,9× speedup mantido. → candidato "v4-lite": trocar randint por
  amostra dos logits no _corrupt_codes (1 forward extra/step).
- Volctrans/WMT21 (arXiv 2109.11247): em finetune, sampling ratio inicial
  0,5 desestabiliza — começar baixo se aplicar.
- WPS experiments: "neighbour corruption" PIORA geração (61–69 vs 43–49
  GenPPL) — corrupção não-uniforme não é automaticamente melhor; a forma
  exata do ruído importa.
- Bengio 2015 (1506.03099) + análise diffio: scheduled sampling viola o
  critério de supervisão correta-por-contexto; nossa corrupção mantém
  targets limpos (denoising), que é o regime correto.

**Decisão: KEEP.** Justificativa: gap ainda descendente/estável na faixa
1,05–1,09, clean CE recuperado — a corrupção está funcionando sem custo
permanente. Se o gap ficar flat em +2 pontos consecutivos (próximos:
@6499, @6999), desenhar braço "self-corruption" (GLAT-lite) warm-start do
melhor ckpt — implementação ~30 linhas em _corrupt_codes + 1 forward/step.

## Braço v5 — intra-level masked conditioning + Mask-Predict (RUN-026)

Hipótese: o resíduo (~+1,1 nats de gap, ~5% dup) é independência condicional
dentro do nível — cada posição amostrada iid. `intra=True` condiciona
posições pendentes aos códigos já decididos do MESMO nível (via mask_emb) e
destrava refinement iterativo estilo Mask-Predict (remascara posições de
menor confiança e re-prediz em paralelo).

Setup: warm-start v3@9499 (snapshot), `--intra --intra-mask mix`
(50% prefixo L→R + 50% subconjunto aleatório por item/nível — cobre decode
chunked e remáscara esparsa), corrupt linear 0.75→0.25 mantido, intra_passes
=32, 2500 steps, OFFSET=9499 → labels 9998..11998+. THREADS=14 (CPU livre).

Custo de decode: chunked intra = ceil(L/32) por nível ≈ 24 passes vs 11 do
paralelo (≈2,2× mais forward passes — o trade-off a medir). Refinement
iters=1, from_level=7 adiciona +10 passes (34 total no sanity).

Sanity (12 steps, batch 4): treino intra roda, warm-start preserva tudo
exceto mask_emb.* (init zero = pending≈paralelo), decode chunked+refine
funciona: passes=34, dup 0,050, gap +0,55 (n=4 — indício apenas).

Comparação pré-registrada: v5@9998 (chunked, refine=0) vs v3@9499 (paralelo)
— e v5@9998 com refine_iters=1 para isolar o efeito do refinement.

## Ciclo v5-1 — série completa + refinement funciona (12 nov)

Treino v5 terminou (2500 steps, ~12,4h). Série chunked intra (refine=0):

| ckpt | dup | dist | self | gt | gap | passes |
|---|---|---|---|---|---|---|
| @9998 | 0.052 | 0.69 | 5.36 | 4.26 | +1.101 | 24 |
| @10498 | 0.055 | 0.59 | 5.37 | 4.30 | +1.079 | 24 |
| @10998 | 0.066 | 0.68 | 5.35 | 4.25 | +1.104 | 24 |
| @11498 | 0.044 | 0.70 | 5.26 | 4.28 | +0.980 | 24 |
| final | 0.053 | 0.67 | 5.13 | 4.22 | **+0.909** | 24 |

vs v3@9499 (paralelo): dup 0.058, gap +1.09, gt 4.22, passes 11.

Leitura:
- gap −0,18 nats e AINDA descendo no final (2 últimos pontos romperam o
  platô ~1.10 que o v3 segurou por 4.500 steps) — intra-level conditioning
  ajuda o modelo a tolerar os próprios códigos.
- dup_rate não melhora (~0,05 nos dois braços) — o stutter residual NÃO é
  primariamente independência intra-nível; é algo mais profundo (l0/arco).
- Custo real: 24 vs 11 passes (2,2×).

**Refinement (final, frac=0.3, from=l7, iters=1):**
dup 0.0453, self 4.858, gt 4.204, **gap +0.655**, passes 29.
→ −0,25 nats sobre o chunked puro em 1 rodada, −40% vs v3. O sinal de
confiança (prob no tempo de predição) está funcionando — o major do blind
review estava certo em exigir a correção.

Em voo: sweep ref2 (iters=2) e ref1-f0.5-from5 — curva dose-resposta.

## Ciclo v5-2 — dose-resposta do refinement: não-monotônico (12 nov)

Sweep no ckpt final (n=8, seed 0):

| config | dup | self | gt | gap | passes |
|---|---|---|---|---|---|
| chunked (refine=0) | 0.053 | 5.127 | 4.217 | +0.909 | 24 |
| it1 f0.3 l7 | 0.045 | 4.858 | 4.204 | +0.655 | 29 |
| it2 f0.3 l7 | 0.054 | 4.777 | 4.225 | +0.552 | 34 |
| it1 f0.5 l5 | 0.047 | 4.667 | 4.251 | +0.416 | 31 |
| it2 f0.5 l1 | 0.088 | 4.047 | 4.273 | **−0.226** | 46 |

**Veredito**: o refinement fecha o gap monotonicamente até INVERTÊ-LO
(−0,23) — mas o dup_rate DOBRA no extremo (0,088). Assinatura de colapso
de modo: o decode converge a códigos "fáceis" (mais previsíveis, mais
repetitivos), não a texto melhor. O gap deixa de ser métrica confiável de
qualidade nesse regime — precisa do par dup/distinct como sanity.

**Ponto ótimo**: refine moderado (it1, f0.3–0.5, l5–l7): gap +0,42–0,66
(≈40–60% do platô v3), dup ~0,045, custo 29–31 passes vs 11 paralelo.

Conclusão do braço v5:
1. intra=True + chunked L→R: gap −0,18 (platô rompido, descendo no fim).
2. Mask-Predict refinement moderado: gap −0,25 a −0,49 adicionais.
3. Refinement agressivo: colapso de modo — primeiro datapoint do teto
   da técnica.
4. dup_rate ~0,05 INVARIANTE em todos os braços → stutter residual é de
   outra natureza (arco global/l0), não independência intra-nível.
5. Custo: 24–31 passes vs 11 — a qualidade do gap compra-se com ~2-3×
   passes. Para TinyStories-word-salad, ainda não há texto coerente; o
   ganho é na robustez de condicionamento (métrica), não na narrativa.
