# v3 — corrupção linear 0.75→0.25 (warm-start v2) — série de avaliação

Braço: `ncm_train_v3`, warm-start `gen_ncm_v2/final.pt`, schedule linear
por nível (0.75→0.25, l0 intocado), 10.000 steps, ckpt rolling a cada 500
(`artifacts/gen_ncm_v3/ckpt.pt`, sobrescrito; snapshots em
`artifacts/v3_eval/`). Lançado ~09 nov 23h UTC; step ~4720 em 10 nov 06:50 UTC.

Protocolo de avaliação (idêntico ao baseline v2): `sample_official.py
--l0 unigram --temp 1.0 --top-k 0 --top-p 1.0 --n 8 --seed 0`
→ dup_rate, distinct_ratio, ce_self_gen vs ce_held_gt (corrupção off).

## Baseline v2 (referência — `bench_l0uni_official.json` / `_exposure.json`)
- dup_rate 0.1095 | distinct 0.6633 | self 5.6481 | gt 4.4124 | gap +1.2357

## Série
| ckpt (~step) | dup_rate | distinct | self CE | gt CE | gap | obs |
|---|---|---|---|---|---|---|
| v3 @4499 (1,5k warm) | 0.0792 | 0.7657 | 5.7923 | 4.5461 | +1.2462 | dup −28% vs v2; gt CE +0.13 (trade-off WPS); gap estável |
| v3 @4999 (2k warm) | 0.0587 | 0.692 | 5.5549 | 4.4908 | +1.0641 | gap começa a fechar; gt CE recuperando |
| v3 @5499 (2,5k warm) | 0.0596 | 0.6775 | 5.4838 | 4.4357 | +1.0481 | dup estável ~0,06; gap −0,19 vs v2; gt CE ≈ baseline |

## Leitura corrente (preliminar — 3 pontos, n=8, ~25% do schedule)
- dup_rate caiu ~pela metade (0,110→0,059) e estabilizou — consistente
  com a hipótese de que corrupção decrescente melhora robustez de geração.
- Exposure gap FECHANDO: +1,24→+1,25→+1,06→+1,05 — o modelo fica menos
  surpreso pelos próprios códigos; mecanismo-alvo da corrupção.
- ce_held_gt recuperou ao baseline (4,436 vs 4,412 v2) — a degradação
  inicial (+0,13) reverteu; trade-off WPS não virou custo permanente.
- self CE caindo monotonicamente (5,79→5,48) em direção ao gt — o gap é
  a métrica a observar nos próximos ~7k steps.
- Amostras: stutter mais leve mas presente; arco narrativo ausente
  (l0 held ~8,0 — problema de escala, não de corrupção).
- Nota: dois watchers rodaram concorrentemente por ~4h (evals duplicados,
  resultados idênticos — determinísticos por seed; benigno).

## Próximos pontos
- @4999, @5499, ... watcher `scripts/watch_v3.sh` (pid em execução)
  avalia automaticamente; resultados em `artifacts/v3_eval/ckpt_step*.json`.
- Veredito ao final (ou convergência): se gap↓ E dup↓ sustentado →
  corrupção decrescente funciona em orçamento pequeno → paper §4.4 vira
  "confirmado em braço próprio"; se gap estável e gt CE degradando →
  trade-off puro, documentar como evidência a favor do limite de escala.
