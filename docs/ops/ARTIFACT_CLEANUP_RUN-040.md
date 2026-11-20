# Manifesto de limpeza de artefatos — RUN-040 (2026-11-16)

## Política de retenção (duradoura)

**NUNCA deletar (registros e insumos):**
- Todo `*.log`, `*.json`, `*.jsonl`, `*.md`, `*.py`, `*.pickle`, métricas,
  configs, amostras (`live_samples.json`), manifests e relatórios `.agentic/`.
- `*.npy` de **códigos/labels/índices** (insumos: codes_*.npy, l0_*.npy,
  kpack_index.npy). Esclarecimento RUN-040/NC-1: a regra é por FUNÇÃO do
  .npy, não pela extensão — dumps de features derivados (feat_*.npy) são
  a classe deletável por excelência.
- `final.pt`/`best` de cada linha de experimento (âncora de reprodutibilidade;
  exceção: runs de profiling/sanity descartáveis, marcados abaixo).

**DELETÁVEIS (pesos regeneráveis/supersedidos):**
- Checkpoints intermediários: `ckpt.pt`, `ckpt_step*.pt` de linhas encerradas.
- Runs descartáveis cujo único propósito era uma medição já registrada
  (profiling, sanity).
- Dumps de features grandes e regeneráveis de experimentos concluídos.

**PROTEGIDOS — run ativo e dependências (verificados em 2026-11-16):**
- `artifacts/ncm_code/` INTEIRO: `gen_code2.pt` (checkpoint ativo),
  `gen_code2.ep3_20261115.pt` (snapshot de rollback), `gen_lessons*.pt`
  (linhagem warm-start), `vq_code_tune*.pt`, `codes_*.npy`, `bpe_code_*`,
  `live_samples.json`, `train_code3.log`, `code_samples_post.json`.
- `artifacts/gen_ncm_v2/final.pt` — fonte de config do LevelGenerator,
  referenciada por `scripts/_gen_acc.py:24` e `scripts/ncm_codetrain.py:170`.
- `artifacts/ncm_official/` — tokenizer/VQ base (`CK=` em ambos scripts).
- `artifacts/convdistill/student_final.pt` — final da linha de destilação.
- `artifacts/checkpoints/vqvae*` — linhagem VQ upstream (658M, seguro barato).
- `artifacts/level_heads_*`, `domcart/`, `codes*/`, `corpus*/` — insumos.

## Inventário e decisão (du antes: artifacts 49G, df /home: 194G livres, 76%)

| Arquivo | Tamanho | Decisão | Justificativa |
|---|---|---|---|
| gen_ar_300m/ckpt.pt | 5466M | DELETAR | intermediário; final.pt preservado |
| gen_ncm_300m/ckpt.pt | 5434M | DELETAR | intermediário; final.pt preservado |
| token_head_qwen05/ckpt.pt | 1228M | DELETAR | intermediário; final.pt preservado |
| token_head_ar_qwen05/ckpt.pt | 1228M | DELETAR | intermediário; final.pt preservado |
| block_heads_qwen05/ckpt.pt | 933M | DELETAR | intermediário; final.pt preservado |
| gen_ar_ts/ckpt.pt | 769M | DELETAR | intermediário; final.pt preservado |
| gen_ar/ckpt.pt | 769M | DELETAR | intermediário; final.pt preservado |
| gen_ncm_ts_run014/ckpt.pt | 744M | DELETAR | intermediário; final.pt preservado |
| gen_ncm_ts/ckpt.pt | 744M | DELETAR | intermediário; final.pt preservado |
| gen_ncm/ckpt.pt | 744M | DELETAR | intermediário; final.pt preservado |
| gen_ncm_intra/ckpt.pt | 735M | DELETAR | intermediário; final.pt preservado |
| gen_ncm_ts2/ckpt.pt | 735M | DELETAR | intermediário; final.pt preservado |
| v3_eval/ckpt_step*.pt (11 arquivos) | ~5272M | DELETAR | sweep de eval encerrado; métricas em ckpt_step*.json preservadas |
| v5_eval/ckpt_step*.pt (6 arquivos) | ~2874M | DELETAR | idem; `final_for_sweep.pt` preservado como final da linha |
| v5ext_eval/ckpt_step*.pt (3 arquivos) | ~1437M | DELETAR | idem (série de eval; jsons preservados) |
| v4_eval/ckpt_step6998.pt | 479M | DELETAR | ponto único de eval; json preservado |
| gen_ncm_v5ext/ckpt.pt | 457M | DELETAR | linha v5ext encerrada; sem final.pt mas métricas preservadas; peso supersedido |
| gen_ncm_v5/ckpt.pt | 457M | DELETAR | intermediário; final.pt preservado |
| gen_ncm_v4/ckpt.pt | 479M | DELETAR | linha v4 encerrada; métricas preservadas |
| gen_ncm_v3/ckpt.pt | 479M | DELETAR | linha v3 encerrada; métricas preservadas |
| prof_compile_on/final.pt | 457M | DELETAR | run descartável de profiling (conclusão: ~0% ganho, registrada) |
| prof_compile_eager/final.pt | 457M | DELETAR | idem |
| sanity_v5/final.pt | 457M | DELETAR | sanity check descartável |
| l0cond/feat_E_train.npy | 635M | DELETAR | dump de features regenerável (RUN-035, resultado negativo registrado) |
| l0cond/feat_H_train.npy | 29M | DELETAR | idem |
| l0cond/feat_E_held.npy | 55M | DELETAR | idem |
| l0cond/feat_H_held.npy | 8.6M | DELETAR | idem |

**Estimativa total a liberar: ~31.4G**

## Resultado (execução T002, 2026-11-16)

- **artifacts/: 49G -> 17G (32G liberados)** — dentro da estimativa (~31.4G)
- Todos os arquivos deletados foram EXATAMENTE os listados na tabela acima;
  nenhum registro/métrica/código removido (jsons de eval preservados ao lado
  de onde estavam os .pt)
- Protegidos verificados íntegros pós-limpeza: gen_code2.pt (479M),
  vq_code_tune.pt (32M), codes_code.npy (25.6M), train_code3.log,
  gen_ncm_v2/final.pt, convdistill/student_final.pt (1.9G)
- Run ativo inalterado: PID 200686 a 709% CPU, log avançando
  (ep1 step 200 loss=3.3537 t=12691s)
- `df /home`: 194G livres; du artifacts: 17G
