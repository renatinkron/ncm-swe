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

## Leitura corrente (preliminar — 15% do schedule, n=8)
- dup_rate na direção da hipótese (corrupção decrescente melhora robustez
  de geração), mas magnitude ainda dentro do plausível de ruído.
- ce_held_gt subiu — consistente com a inversão CE↔geração da réplica WPS:
  corrupção forte piora CE limpo ao preparar o modelo p/ próprios erros.
- Exposure gap inalterado — o efeito principal, se houver, deve aparecer
  nos checkpoints seguintes conforme o schedule fortalece a representação.
- Amostras: ainda stutter visível; nenhuma melhora qualitativa de arco.

## Próximos pontos
- @4999, @5499, ... watcher `scripts/watch_v3.sh` (pid em execução)
  avalia automaticamente; resultados em `artifacts/v3_eval/ckpt_step*.json`.
- Veredito ao final (ou convergência): se gap↓ E dup↓ sustentado →
  corrupção decrescente funciona em orçamento pequeno → paper §4.4 vira
  "confirmado em braço próprio"; se gap estável e gt CE degradando →
  trade-off puro, documentar como evidência a favor do limite de escala.
