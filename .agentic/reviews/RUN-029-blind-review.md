# RUN-029 blind review — decode followup

Escopo frio: diff de `generator.py` + `sample_official.py` vs HEAD~1.

## Verificações

- Head windowing `h[:, 1+s:1+s+ip]`: mesmas posições que antes eram
  fatiadas depois do head — linear per-posição torna a operação exata;
  w derivado de `codes_s.shape[1]` trata fim de chunk. conf gather
  inalterado. Refine path mantém head cheio (remáscara é por topk
  espalhado — não fatiável; necessário).
- `_chunk_trunk`: extração fiel do loop de `_forward_chunk` (blocos +
  ln_f), assinatura explícita de máscara/rope — semântica preservada.
- `enable_decode_compile`: smoke test B=1 T=129; runtime B=8 dispara
  recompile dinâmico (medido correto); exceção → `_decode_trunk_fn=None`
  → eager silencioso. Opt-in via flag — default inalterado.
- Warmup contado dentro do primeiro generate: honesto — reportado
  separadamente (frio 19,77 vs quente 7,4s).
- l0 path (1 posição) permanece eager — correto, custo trivial.
- Sem mudança em RNG stream, contagem de passes, ou ordem de sampling.

## Achados

- nit: `enable_decode_compile` compila smoke em B=1 e runtime usa B≥1 —
  recompile extra no primeiro passe real; já contado no warmup reportado.
- nit: refine path poderia futuramente fatiar head por índice gather —
  fora de escopo, ganho marginal.

0 blockers, 0 majors, 2 nits. Aprovado.
