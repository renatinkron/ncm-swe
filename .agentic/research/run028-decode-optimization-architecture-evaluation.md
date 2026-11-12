# RUN-028 architecture evaluation — chunk-local decode

## Estado e fluxo

Fato estrutural (block-diagonal + condicionamento via embedding): o output
do chunk k depende SÓ de [q0 + chunk k]. O decode atual roda trunk sobre
[q0 + chunks 1..k] — prefixo morto ~4–5×. Validado bit-idêntico na
pesquisa (k2/k5 diff 0.0; argmax 100% em todos os níveis).

## Decisões

### D1 — novo `_forward_chunk` substitui o caminho interno do generate
- `_forward_partial` permanece intacto (referência/retrocompat — usado
  por ferramentas e sanity tests), `generate()` passa a usar chunk-local.
- Assinatura: `_forward_chunk(base_chunk, k_next, cur=None, decided=None)`
  onde `base_chunk` (B,L,D_vq) já é o input VQ do nível; intra injeta
  `where(decided, emb(cur), mask_emb)` por passe — igual hoje.

### D2 — indexação GLOBAL é o ponto crítico de correção
RoPE e pos/lvl embeddings são absolutos: posição j do chunk k tem pos
global `gofs = 1 + Σ_{i<k} L_i`. Erro sutil: indexar localmente altera
distância q0↔chunk → logits diferentes. Implementação: `pos_ids =
[0] + range(gofs, gofs+L)`; `rope_cos/sin[pos_ids]`; `pos.weight[gofs:..]`;
`lvl.weight[k]`. Máscara local (1+L)²: linha 0 self-only, resto livre.

### D3 — z_hat incremental no loop de níveis
`generate()` mantém `z_hat_acc` acumulado; ao fechar o nível k, soma a
contribuição do nível k (emb+up+resi — mesma lógica de `_up_to_full`/
`_apply_quant_resi`) e faz 1 interp para o chunk do nível k+1. Elimina o
rebuild O(K) por passe (upstream `get_next_autoregressive_input` parity).
Cuidado: embeddings/interp são congelados — cachear em `self` não polui
state_dict (plain attr).

### D4 — q0 KV constante
q0 input é fixo por modelo → suas K/V em cada bloco são constantes por
generate(). Micro-ganho (<2%): implementar como detalhe opcional só se
trivial; não bloqueia.

### D5 — caminho não-intra também migra
v2/v3 (intra=False) usam o mesmo loop → mesma otimização ~4×: cada nível
vira [q0 + chunk]. Cobre os ckpts antigos sem flag.

### D6 — decode adaptativo por confiança NÃO entra nesta implementação
Fast-dLLM-style adaptive commit muda a dinâmica de amostragem → é
experimento científico próprio (braço de eval), não refactoring. A
arquitetura chunk-local o destrava como flag futura.

### D7 — trust boundaries
- Numérica: diffs ≤5,7e-6 medidos (tiling fp); sampling T=1.0 pode
  divergir ocasionalmente → métricas dentro do ruído, tokens não
  bit-idênticos — aceitável e documentado.
- Semântica: logits-level, não token-level — prova por comparação.

## Critérios de validação
1. `_forward_chunk` vs `_forward_partial`: logits max_abs_diff < 1e-4 e
   argmax 100% em k∈{2,5,8,11}, B∈{1,8}, ckpt v5 (validado em pesquisa —
   re-verificar após integrar)
2. Eval seed 0 n=8 receita oficial: dup ±0,01, gap ±0,05 vs JSON v5 final
3. gen_s B=8: ≥3× mais rápido no mesmo ckpt
4. ckpt não-intra (v3 final): eval funciona e acelera
5. refine_iters>0 com chunk-local: passes idênticos, mesma dinâmica
