# RUN-041 blind review — changed files cold

Arquivos: `scripts/_constr_decode.py` (novo), `SESSION_HANDOFF.md` (append).

## Achados

1. **`viable()` aceita code/None de `compile_command` — oráculo explorável**
   — confirmado pelo próprio experimento: tokens que abrem `#`/`"""` tornam
   qualquer sufixo viável; hits "reais" foram 1 stmt + filler. Severidade:
   limitação de método documentada no handoff, não bug. Se o script for
   evoluído, oráculo precisa banir entrada em comment/string ou pontuar
   só no parse final.

2. **`torch.tensor(codes_row)` em gen path recebe lista python — OK**
   `.long()` aplicado cobre o caso uint16 do numpy (bug real encontrado e
   corrigido em iteração). Sem problema residual.

3. **`eval_rows` mistura `try/ast.parse` com `n_real` — consistente**
   `n_real` retorna -1 em falha de parse, então `nr` só conta parseáveis
   com stmt real. Correto.

4. **Vazamento: nenhum** — script read-only em .pt, grava só jsons de saída.
   Paths hardcoded /repo + /work/artifacts consistentes com irmãos
   (_sample_parse.py, ncm_codetrain.py). Sem rede, sem deps novas.

5. **Menor**: `np_` conta parse incluindo degenerados — porém a métrica
   `real` separa isso parcialmente; o caveat está documentado. Aceitável
   para um script de medição descartável.

6. **Menor**: `SyntaxWarning` do interpretador pode poluir stdout — já
   observado no run; não afeta JSON.

## Veredito

Sem bloqueadores. Script de medição isolado, honesto nos resultados,
sem efeitos colaterais fora de artifacts/ncm_code/*.json.
