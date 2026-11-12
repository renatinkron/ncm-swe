# RUN-027 blind review — diff frio (generator.py, train_gen_official.py)

Revisado sem contexto do spec: diff b7a6a70..HEAD nos arquivos de produção
(profile_*.py são ferramentas descartáveis — fora do escopo crítico).

## Majors: 0 | Blockers: 0

Verificações frias:
- `_trunk_fn` é atributo plano (não Parameter/buffer/Module) → fora do
  state_dict; `load_state_dict(strict=True)` e `torch.save` inalterados ✓
- `_trunk()` reproduz a semântica original bit-a-bit (mask slice por
  h.shape[1], rope slices, ln_f final na mesma ordem) — confirmado
  empiricamente por losses idênticos nos 20 steps ✓
- `enable_trunk_compile` captura falha no smoke call (inductor é lazy —
  a chamada é onde BackendCompilerFailed sobe) → fallback correto ✓
- `torch.compile(self._trunk)` não substitui parameters → optimizer criado
  depois continua vendo os mesmos objetos ✓
- Eval/generation inalterados: `_forward_partial` mantém loop próprio;
  `_trunk_fn` default None → `loss()` eager em sample_official ✓
- `except Exception` cobre AttributeError em torch<2.0 → retorna False ✓

## Nits (não-bloqueantes)
1. Smoke test usa B=1 — primeira chamada real (B=16) recompila (~45s
   extra, uma vez por run). Alternativa: dummy no batch real, mas o
   modelo não conhece B — custo/benefício não fecha.
2. `profile_kernel_alts.py`: variável `qs` criada e não usada (linha 60)
   — ferramenta descartável, não-corrigida.

## Veredito
Clean — nenhuma ação obrigatória.
