# Blind Review — RUN-048

**Escopo**: changed/new files, sem spec/task_plan. Arquivos:
`scripts/ncm_panel.py`, `scripts/dump_knowledge.py`,
`data/lora_dump/knowledge_dump.jsonl`, `tests/test_panel.py`,
`tests/test_dump.py`, `docs/ncm-render-panel.md`.

## Findings

### B1 — `--out` relativo corrompido por `os.chdir` (MAJOR → corrigido)
`load_panel()` executa `os.chdir(vendor/ncm-upstream)`; qualquer `--out`
relativo passado depois resolvia dentro do diretório vendor. Corrigido:
`args.out = os.path.abspath(args.out)` antes do load. Smoke usou path
absoluto, então não explodiu em produção — mas era armadilha real.

### B2 — `--l0 model` caía em assert cru do upstream (MINOR → corrigido)
No ckpt oficial (`predict_level0=false`, `query_level0=false`), `--l0 model`
produzia `AssertionError` interno em vez de mensagem clara. Agora
`SystemExit` explica o requisito. Também corrigi a mensagem do branch
`l0_p is None`, que citava "--l0 model" no lugar errado.

### B3 — `l0_source(generator=)` parâmetro morto (NIT)
`torch.multinomial` usa o RNG global; o parâmetro `generator` nunca é usado.
Inócuo (seed global já fixado), mas é API enganosa — mantido por compat com
assinatura documentada; remover na próxima limpeza.

### B4 — `allow` para `--mu` é `torch.ones(V)` (NIT, documentado)
`eval_shard` usa vetor `ALLOW` do config; painel usa ones = "todo token pode
ser repetido T vezes". Comportamento razoável como default; documentar no
`--help` quando `--mu` for usado de verdade.

### B5 — cobertura de teste do painel é helper-level apenas (INFO)
Nenhum teste exercita `load_panel`/`main` (exigiria ckpt); decisão correta
para suite rápida — o smoke manual em container cobre o caminho real.

## Veredito
0 blockers, 0 majors abertos (B1/B2 corrigidos e re-testados: selftest OK,
21 passed). B3–B5 são nits/informativos.
