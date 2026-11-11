# RUN-009 Blind Review

Cold review: `scripts/codepath_decode.py` (new), `spec_pyramid.py`
(--score path), `artifacts/report.md` (seção), eval/bench JSONs.

## Findings

- **info** — `codepath_decode.py`: `truth` (token real do corpus)
  gravado por posição mas não agregado no summary — acordo medido é
  contra o argmax AR, escolha documentada e adequada (o baseline
  prático), mas o dado de verdade existe no per_pos para análise
  posterior.
- **info** — `path_scores` beam: o topk é feito uma única vez na
  primeira passada por `k >= beam_after` (podado permanece podado) —
  semântica de beam consistente; re-poda por nível seria variante, não
  bug.
- **info** — modo `full`: custo ~30s+/pos no nível-5 (152k×256×16384)
  inviabiliza eval grande em CPU — o report/documentos registram isso
  e o beam (0,7s/pos) é a variante usada; limitação honesta, não
  funcional.
- **verified ok** — `score(v)` usa `tok2codes[v, k]` (códigos do
  PRÓPRIO candidato em todos os níveis) — exatamente a semântica de
  teacher forcing do treino; nenhuma mistura com códigos amostrados.
- **verified ok** — `--score path` no spec_pyramid chama path_scores
  por offset head; `equal_ar=True` preservado nos runs (correção
  estrutural intacta).
- **verified ok** — números do report.md batem com
  `codepath_eval.json` (0/40 ambos métodos) e o diagnóstico de rank
  (~43k mediano) foi executado em script inline — evidência registrada.
- **verified ok** — `tok2codes.pt` cacheado; rebuild determinístico.

## Patterns

n/a — achados info-level isolados.

## Verdict

Sem blockers, sem majors. Resultado negativo robustamente medido
(3 decoders independentes, mesmo 0%) e corretamente atribuído à
qualidade das distribuições de código, não ao método de decode.
