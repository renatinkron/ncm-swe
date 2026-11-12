# RUN-030 blind review — v5-ext + stats

Escopo frio: `scripts/watch_v5ext.sh`, `scripts/eval_ckpt_v5ext.sh`
(derivados via sed dos v5), runner de stats (/tmp, host-side).

## Verificações

- OFFSET=12000 consistente nos dois scripts; FIRST=12498 → primeiro
  boundary computado 12499 (local 499 + offset) — bate com
  `step%500==499` do trainer e com `--offset 12000`.
- Dirs novos `gen_ncm_v5ext`/`v5ext_eval` — não sobrescreve artefatos
  da série v5; snapshots nomeados por step global.
- Guard anti-torn-snapshot (mtime ≥120s + re-check) preservado.
- Stats runner: mesma receita oficial (T=1.0, k=0, p=1.0, l0 unigram),
  n=32, seeds 0/1/2; refine só no braço correspondente; `continue`
  evita re-avaliar JSONs existentes.
- Treino: `--init-from` no ckpt correto, offset consistente,
  `--compile-trunk` já validado bit-idêntico (RUN-027).
- v3_9499 usa snapshot `v3_eval/ckpt_step9499.pt` (o braço terminou em
  parada antecipada — ckpt correto do ponto final avaliado).

## Achados

- nit: watcher v5ext herda MAXSEC=72h — sobra (~11h previstos).
- nit: stats runner em /tmp não versionado — efêmero por desenho;
  JSONs e aggregate.json são o artefato durável.
- nit: contention treino×eval afeta s_per_sample dos JSONs de stats —
  documentado no execution report; métricas de qualidade imunes.

0 blockers, 0 majors, 3 nits. Aprovado.
