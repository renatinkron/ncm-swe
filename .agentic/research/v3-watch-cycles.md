# v3 — ciclos de monitoramento 2h (RUN-025)

Cadência: a cada ~2h (ou a cada ckpt novo — evals caem ~2,5h): colher ponto,
pesquisa direcionada à pergunta que os dados levantam, decidir
keep/adjust/kill. Gatilhos de decisão no plano RUN-025.

Baseline: v2 dup 0.1095, gap +1.2357, gt 4.4124.

| ciclo | hora | último ponto | dup | gap | decisão |
|---|---|---|---|---|---|
| 1 | 10 nov 16:45 | @5999 | 0.0542 | +1.089 | **KEEP** — ver análise |

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
