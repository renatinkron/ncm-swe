# RUN-028 blind review — chunk-local decode

Escopo frio: `src/ncm_swe/model/generator.py` (diff vs HEAD~1) + scripts de
profiling novos. Sem contexto de spec na revisão; semântica verificada
contra `_chunks_from_codes`/`_forward_partial` do próprio arquivo.

## Verificações estruturais feitas

- `_gofs[k] = 1 + sum(levels[1:k])`: offsets conferem com o layout
  `[q0]+chunks` de treino (`attn_mask`/`lvl_ids`).
- Mask (1+L)×(1+L): chunk rows all-zero (veem q0 + chunk inteiro) =
  block-diagonal + q0 visível, idêntico ao que `_forward_partial`
  produzia para as posições lidas. Linha q0 com chunk bloqueado é
  semântica-neutra: logits consumidos vêm só de `h[:,1:]`; K/V do q0
  derivam do input embedding, não da sua linha de saída.
- RoPE/pos/lvl indexados por posição global (RoPE absoluta — correto;
  indexar local mudaria distâncias q0↔chunk vs treino).
- `_level_contrib`/`_interp_chunk`: ordem emb→up_to_full→quant_resi→
  accumulate→interp idêntica a `_chunks_from_codes` (incl. exceção do
  último nível sem up_to_full — nunca alcançada, guard `k+1<len`).
- Intra: mask_emb injetado onde `decided=False`, emb(code) onde True —
  mesmo padrão de `_forward_partial`. Refine rounds usam `decided2`
  sem alterar `decided` — protocolo preservado.
- Ordem de sampling/RNG inalterada (mesmo nº de `_sample` calls, mesmas
  shapes) — corroborado por textos byte-idênticos no eval seed 0.
- `_forward_partial` inalterado, disponível como referência.
- `_gofs` é attr plain (não-buffer): recomputado no ctor, não afeta
  checkpoints.

## Achados

- nit: `_forward_chunk` aloca `m`/`pos_ids`/`cos`/`sin` por chamada
  (~30+/gen). Pré-computar por nível pouparia micro-allocations —
  desnecessário a 6.79× de speedup.
- nit: `sum(self.levels[1:k])` poderia usar `_gofs`; já refatorado via
  `_gofs[k]` no código final — sem ação.
- nit: scripts `profile_decode_*.py` são dev-tools sem CLI guard
  elaborado; executados só manualmente em container.

0 blockers, 0 majors, 3 nits informativos. Diff aprovado.
