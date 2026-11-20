# Blind Review — RUN-040

**Escopo revisado (a frio, sem spec)**: `scripts/_gen_acc.py` (parametrização
env), `docs/ops/ARTIFACT_CLEANUP_RUN-040.md`, `course/README.md` +
`course/modules/m00..m10`, `SESSION_HANDOFF.md` (seções novas), deleções de
artefatos, `.agentic/state/runs/RUN-040.*`.

## Achados

**Nenhum bloqueador, nenhum major.**

### Minors/observações (não exigem ação)

1. `scripts/_gen_acc.py` é não-rastreado (??) — script diagnóstico;
   consistente com o padrão de outros `_*.py` no repo. Mudança é
   retrocompatível: defaults de `GEN_CK`/`CODES`/`OFFSET` preservam o
   comportamento anterior.
2. Números do curso conferem contra artefatos primários:
   bpe_metrics.json (8192 vocab, 2.75 ch/tok), codes_code.npy
   25.640.128B = 20000×641×2B ✓, gen_code2.pt 479.267.786B ≈ 119.8M
   params fp32 ✓, batch 16×641=10256 ✓.
3. Módulos citam exclusivamente medições presentes no handoff ou em
   metrics — sem número inventado detectado.
4. Manifesto corrige prospectivamente a redação "*.npy" (NC-1) — boa
   forma: regra passa a ser por função, não por extensão.
5. Deleções conferem com o manifesto; nenhum registro/métrica removido;
   dependências ativas (gen_ncm_v2/final.pt referenciado por
   `_gen_acc.py:24` e `ncm_codetrain.py:170`) preservadas — o ponto mais
   perigoso da limpeza foi coberto corretamente.
6. README do curso inclui o disclaimer de "em construção, lançamento ao
   fim do caminho probatório" — alinhado com a instrução do usuário.

## Veredito

CLEAN — sem non_conformities adicionais a introduzir.
