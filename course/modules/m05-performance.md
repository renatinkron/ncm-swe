# M05 — Performance: profile antes de otimizar

**Objetivo**: nunca otimizar por intuição — medir onde o tempo vai,
estimar o teto teórico, e só então escolher a alavanca.

## A hipótese que o profile refutou

Pergunta legítima do operador: *"PyTorch é genérico — será que não está
prejudicando o treino do NCM?"*

Em vez de debater, medimos. `py-spy` no processo vivo, 60s, 5.939 amostras:

```
65.4%  autograd backward        ← engine nativa C++
24.3%  Linear.forward (GEMM)    ← matmul nativo MKL/oneDNN
 3.1%  SelfAttention
 1.5%  RMSNorm
 ~2%   todo o resto junto — incluindo Python puro
```

**~98% do tempo já estava em código nativo.** A "interpretação Python"
custava frações de milissegundo por step de 15s. A hipótese de overhead
de framework morreu com dados — e a featurização VQ suspeita nem apareceu
no radar (<0.3%).

## A matemática que previa o resultado — roofline

Antes de qualquer otimização, calcule o teto do hardware:

```
Modelo: ~120M params fp32, batch 16×641 = 10.256 tokens/step
FLOPs/step ≈ 6 × params × tokens ≈ 7.4 TFLOP
A ~15.5s/step → ~470 GFLOPS sustentados

Pico Zen2 (8 núcleos, 2×FMA AVX2, ~3.2-3.5GHz all-core):
  8 × 16 FLOP/ciclo × 3.3GHz ≈ 420-490 GFLOPS teóricos

=> treino já roda a ~95-100% do roofline fp32 da CPU
```

Quando você está no teto do silício, **nenhuma biblioteca dá mais do que
migalhas**. Trocar BLAS, allocator ou framework não compra FLOPs que o
hardware não tem.

## O experimento controlado — três otimizações, três vereditos

| Otimização | Ganho medido | Lição |
|---|---|---|
| `torch.compile` (Inductor C++) | ~0% (15.3 vs 15.5s/step) | GEMM já vai pro MKL; fusão de ~2% pointwise não move ponteiro. Custo: 14min de warmup + bug de stride no VQ |
| affinity 8 núcleos físicos | **mesmo ritmo com menos CPU** | SMT não ajuda GEMM (FPU compartilhada); pinning libera ~4 threads pro desktop permanentemente |
| tcmalloc | ~nulo | ajuda workloads de alocação pequena; o nosso é GEMM-bound |

Nota: `torch.compile` não foi descartado por falha — foi medido e empatou.
A decisão de mantê-lo ou não vira pragmática (warmup, complexidade), não
esperança.

## Por que não Rust/C?

A pergunta correta era: "as bibliotecas nativas por baixo têm propostas
melhores?" Resposta medida: oneDNN/MKL 2024 já despacham AVX2/FMA no Zen2;
ZenDNN/AOCL dariam 0-10% em GEMM denso já saturado. Reescrever o treino
em Rust/C trocaria o maestro (Python, <2%) — o orquestrado (matmul nativo)
ficaria igual. **Atacaria os 2% errados.**

## Lab 5 — Faça a conta antes de propor

**Objetivo**: estimar roofline do SEU workload antes de sugerir otimização.

```bash
# O que a CPU suporta:
lscpu | grep -iE "model name|flags" | grep -oE "avx2|avx512|fma|amx" | sort -u
# Onde o tempo vai de verdade (processo vivo):
py-spy top --pid <pid_do_treino> --duration 30
```

Conta: params × tokens/step × 6 ÷ s/step = GFLOPS. Compare com
`núcleos_físicos × 16 FLOP/ciclo × GHz all-core`.

**Observe**: a fração em Python puro vs nativo no profile.

**Decida**: se o seu número já está ≥80% do roofline, escreva a frase que
mata a discussão de "trocar de framework". (A do projeto: *"o próximo
multiplicador de velocidade é GPU ou modelo menor — em CPU estamos no
silício"*.) 

---

*Anterior: [M04](m04-metricas.md) · Próximo: [M06 — Governança de recursos](m06-recursos.md)*
