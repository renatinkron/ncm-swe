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
