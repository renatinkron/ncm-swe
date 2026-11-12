# RUN-027 research — profiling do treino NCM + espaço de otimização

Demanda: pesquisa profunda de performance do treino/inferência — reescrita
C/Rust, cálculos mais eficientes, implementações obsoletas, máxima performance.

## Reuse check

- Nenhuma biblioteca externa de treino substitui o stack atual para este caso:
  PyTorch CPU já delega os GEMMs ao MKL (verificado: `mkl: True`).
  JAX/XLA-CPU usa o mesmo oneDNN — mesma ordem de grandeza, custo de port alto.
- Kernel libraries (oneDNN/MKL) já estão sendo usadas pelo dispatcher do
  PyTorch 2.5.1 — medido: ~84% do pico teórico da CPU.
- Para inferência: ggml/llama.cpp-style C engine ou candle/burn (Rust) —
  relevante só no decode (ver §inferência). Não aplicável ao treino.

## Setup medido

- CPU: Ryzen 7 PRO 4750U (Zen 2, 8c/16t, AVX2 apenas — sem AVX-512/VNNI/BF16/AMX)
- torch 2.5.1+cpu, MKL on, imagem `localhost/lcm-infer` (e `lcm-infer-gcc` p/ compile)
- Modelo: LevelGenerator 64,35M treináveis (100,8M c/ tokenizer congelado no grafo),
  4 blocos, embd 896, GQA 14q/2kv, SwiGLU 4864, T=641 (q0+11 chunks), B=16
- Braço perfilado: config v5 real — intra=mix, corrupção linear 0,75→0,25,
  warm-start `gen_ncm_v5/final.pt`
- Scripts: `scripts/profile_train.py`, `profile_variants.py`,
  `profile_kernel_alts.py`; JSONs em `artifacts/prof/`

## Onde o tempo vai (medido, ~17,3s/step @ B=16/8 threads)

Fases (8 steps):
- forward ≈ 6–6,5s (~37%)
- backward ≈ 10,2s (~59%)
- optimizer (clip+AdamW) ≈ 0,23s (~1,4%)
- head+loss ≈ 0,03s; data pipeline ≈ 0,001s — irrisórios

Ops por step (torch.profiler, 4 steps):
| op | self ms/step | calls | o que é |
|---|---|---|---|
| aten::mm | 7 558 | 79 | GEMMs do backward (dX, dW) |
| aten::addmm | 3 417 | 20 | Linears do forward (4 blocos × 5) |
| sdpa_cpu bwd | 841 | 4 | flash-attention-for-cpu backward |
| aten::mul | 839 | 211 | RoPE, SiLU*up, máscaras de loss |
| aten::copy_ | 451 | 815 | repeat_interleave GQA, transposes, cats |
| sdpa_cpu fwd | 350 | 4 | flash-attention-for-cpu forward |
| fill_/add_/sum/silu/cat/div/pow | ~900 | ~900 | elementwise + reductions |
| mkldnn_convolution | 41 | 11 | `F.interpolate` (idx_to_ncm_input) |

**GEMM total ≈ 11,0s ≈ 65–70% do step** — já é MKL puro.
SDPA ≈ 1,2s (~7%). Elementwise/copies ≈ 2,5–3s (~17%). Resto ≈ autograd/loss/opt.

## Roofline

- GEMM medido: 355–372 GFLOPS (shapes 10256×896×4864 / ×1152)
- Pico teórico Zen2 @~3,4GHz boost, 8 cores, 2×FMA-256: ~435 GFLOPS
- → **MKL entrega ~84% do pico** — o gargalo é o volume de FLOPs, não a
  implementação. Treino é compute-bound, não overhead-bound nem memory-bound.

## Experimentos A/B (todos medidos)

| variante | resultado | veredito |
|---|---|---|
| threads 16 (SMT) vs 8 | 15,17 vs 14,84s | neutro/levemente pior — 8 ótimo |
| batch 32 vs 16 | 1083,9 vs 1079,6 ms/amostra | nulo — GEMM já saturado |
| SDPA `enable_gqa` vs repeat_interleave | 853 vs 327 ms (fwd+bwd) | **2,6× pior** — kernel CPU 2.5.1 não otimiza GQA |
| atenção per-level (11 SDPA + concat) | 596 vs 327 ms | **1,8× pior** — overhead de grafo > FLOPs economizados |
| `F.rms_norm` vs RMSNorm python | 78,8 vs 79,6 ms | nulo |
| torch.compile forward inteiro | AssertionError stride (inductor) | quebrado no torch 2.5.1 |
| **torch.compile trunk-only (4 blocos)** | **11 856 vs 13 807 ms** | **−14% no trunk ≈ ~10–12% e2e** |

Nota: o microbench fwd-only sugeria per-level 3,9× mais rápido (81→21ms), mas
com backward + concat + setup de grafo vira 1,8× mais lento — lição: medir
fwd+bwd sempre; sparse-via-N-chamadas não compensa sem kernel nativo.

## Veredito C/Rust p/ treino

**Não paga.** ~67% do step já é MKL a ~84% do pico do hardware. Uma reescrita
não melhora GEMM; o que sobra é autograd+elementwise (~25%), que só um
compilador de grafo (inductor) ou kernels fundidos à mão capturariam — e mesmo
assim o teto realista é ~1,15–1,3×.

## Alavancas reais, ordenadas por ROI

1. **torch.compile no trunk** (imagem com g++): ~10–12% e2e medido. Único
   ganho >5% encontrado. Integração: compilar wrapper dos 4 blocos no trainer.
2. **Menos FLOPs** (única alavanca de ordem maior): modelo menor, T menor,
   ou menos passes — decisão científica, não engenharia.
3. **Hardware**: AVX-512/VNNI/BF16/AMX dobrariam-triplicariam o pico; fora de
   escopo (Zen2 não tem).
4. Descartados com dados: batch maior, SMT, GQA-SDPA, atenção por nível,
   rms_norm fundido, reescrita C/Rust.

## Inferência (contexto: claim de velocidade do paper)

Decode B=1 é memory-bound + dispatch-bound, não GEMM-bound:
- ~190–450ms/passe observado vs ~10–30ms roofline (256MB pesos, ~30GB/s)
- `_forward_partial` recomputa o prefixo congelado a cada chunk-passe intra —
  com intra_passes=32 e L=128, até 4 passes redundantes/nível; prefix caching
  (past_kv já existe na API do Block) eliminaria ~2–4× do custo intra/refine
- eval n=8 sequencial: batching amortizaria ~8×
- int8 (ggml-style AVX2 dot) + engine C/Rust zero-dispatch: plausível 2–4×
  adicional — único lugar onde reescrita em C/Rust se justifica
- Estimativa combinada honesta: 5–15× no decode, vs ~1,15× no treino

## Riscos/observações

- Números de step variam ~±15% entre runs (contenção/turbo); comparar dentro
  do mesmo run.
- compile do forward inteiro quebra no head loop (stride assert) — trunk-only
  é o recorte funcional em torch 2.5.1; retestar em versão futura.
- lcm-infer-gcc existe só localmente (derivada de lcm-infer + g++).
