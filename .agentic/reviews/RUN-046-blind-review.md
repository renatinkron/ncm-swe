# Blind Review — RUN-046

**Changed files**: `scripts/_func_units.py`, `scripts/_bpe_probe.py`, `scripts/_eval_bpe_prefix.py`
**Adjacent**: `scripts/ncm_codetrain.py` (padrão LESSONS/cache), `artifacts/ncm_code/*` (consumo)

## Findings

| file | line | severity | issue |
|---|---|---|---|
| _func_units.py | 59,78 | minor | stat `fit127` mantém nome literal "127" mas conta unidades ≤ `MAX_TOK` — com `MAX_TOK=255` o campo fica semanticamente errado no stats JSON gravado |
| _func_units.py | 98 | minor | `OUT_NAME.replace(".txt", "_stats.json")`: se `OUT_NAME` não contiver `.txt`, replace é no-op e o stats JSON é gravado **no mesmo path** das unidades recém-escritas (sobrescreve o arquivo de dados). Env-gated, default seguro — edge case silencioso |
| _bpe_probe.py | 137-140 | info | `decode()` usa `ids.index(0)` — válido para listas de `generate()`, mas quebraria em np.ndarray (`ndarray` não tem `.index`); uso atual só recebe listas — latente, não ativo |
| _bpe_probe.py | 109 | info | BOS e EOS compartilham token 0 — design deliberado e medido (parse 26/32), mas acopla early-stop e início de sequência; se o modelo aprender EOS prematuro em contexto longo, o efeito aparece só em ctx>256 |
| _bpe_probe.py | 198-207 | info | `INIT` seguido de `EVAL_ONLY`: se ambos setados, o load do INIT é redundante (EVAL_ONLY recarrega CKPT). Sem erro funcional, só trabalho morto |
| _bpe_probe.py | 115 | info | cache antigo `bpe_rows.npy`/`bpe_units_rows.npy` (sem sufixo ctx) ficou órfão em disco — não é mais lido; só clutter |
| _eval_bpe_prefix.py | 53 | info | chaves do metrics mudaram `def_prefix_*` → `prefix_*` — drift de schema vs arquivos históricos; aceitável por incluir `ckpt`/`prefix`/`bos` provenance |
| _eval_bpe_prefix.py | 51 | info | strip `r[1:] if r[0]==0` remove corretamente o BOS; edge BOS=0 com primeiro token gerado=0 é no-op inócuo (empty→empty) |

## Patterns

Nenhum padrão com 3+ achados de mesma raiz.

## Trajectory

Eventos do run consistentes com os artefatos: mid-eval (`bpe256mid` container, `bpe_func256_mid.json`) e re-medição pós-fix (`gen_bpe_func256_prefix_metrics.json` + texts JSON) existem em disco — claims de execução têm evidência material. O incidente de eval (BOS-prefix→decode vazio) está documentado no execution_report e visível na divergência parse 32/real 0 da rodada bugada vs 26/26 da corrigida.

## Verdict

Código pequeno, env-gated e defaults preservados em todos os arquivos — critérios de contrato respeitados. Achados: 2 `minor` (stat `fit127` com nome defasado; edge-case `OUT_NAME` sem `.txt` sobrescrevendo stats), resto `info`/latente. Nenhum `major`/`blocker`. O bug real introduzido no run (BOS no prefixo decodificado) já foi detectado, corrigido e re-medido dentro do próprio run — não resta débito funcional.
