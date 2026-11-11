# RUN-011 Blind Review

Escopo: mudanças da run = `artifacts/report.md` (seção RUN-011),
`token_head_eval2.json`, `spec_pyramid_tokhead_bench2.json`,
`token_head_qwen05/final.pt`+`ckpt.pt` (binários). Nenhum arquivo de
código alterado (eval foi script one-off em /tmp/th_eval.py).

## Revisão dos artefatos

1. **Eval protocol** — o script de eval (inspecionado em /tmp):
   `hold` = últimos 5% do stream (split idêntico ao do trainer),
   128 posições = primeiras 128 linhas do hold, batches de 32;
   `feat = model(inp)[:, -1]` idêntico ao train; `predicted_code_vecs`
   reutilizado do módulo (sem reimplementação divergente); `accs` do
   módulo. Sem indexação manual nova → sem risco da família de bugs
   off-by-one corrigidos antes. Protocolo consistente com o documentado.

2. **Ressalva honesta** — o comando exato do eval RUN-010 não foi
   preservado em arquivo; assume-se que era o mesmo split/posições
   (leitura natural do "128 posições held-out" documentado). O
   compliance_report registra isso explicitamente. Impacto: baixo —
   a diferença 3 vs 4 acertos está dentro de ±1 (ruído de contagem),
   e a conclusão (estagnou) é suportada independentemente pela curva
   de loss (todos os top-5 evals no treino original).

3. **Spec bench** — rerun do binário já revisado (spec_pyramid.py,
   revisado na RUN-007/010) com `--score tokhead` sobre o novo ckpt.
   Config registrada no JSON; equal_ar=True mantido.

4. **Report.md** — a seção RUN-011 reporta regressão leve sem
   suavizar; conclusão "gargalo estrutural" é suportada pelos
   números citados (verificados contra os JSONs).

5. **Sem issues de segurança** — sem código novo, sem deps, sem
   segredos; binários de checkpoint produzidos pelo trainer já
   revisado.

## Findings

- Nenhum major/minor funcional. Observação processual: persistir o
  script de eval dentro do repo (ex.: `scripts/eval_token_head.py`)
  removeria a ressalva do item 2 em runs futuras — recomendação
  não-bloqueante.

verdict: clean
