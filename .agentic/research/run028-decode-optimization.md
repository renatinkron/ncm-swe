# RUN-028 research — otimização do decode NCM

Demanda: pesquisa sobre o decode — verificar se pode ser melhorado.
Conclusão: **sim, ~5× medido em PyTorch puro, bit-idêntico** — e é a
mesma estrutura que o código oficial já usa.

## Reuse check — o upstream JÁ resolveu isso

`vendor/ncm-upstream/models/next_concept.py` `generate()` (linhas 660–694):
- roda **só o chunk do nível atual** pelo trunk
  (`x = wte(chunk) + level_pos_embed[global]`), nunca o prefixo de códigos
- `use_kv_cache` cacheia apenas o prefixo de TEXTO (contexto), não chunks
- `get_next_autoregressive_input` acumula z_hat **incrementalmente**
- oferece `attn_pattern` block_diagonal/block_causal + flex_attention

Nosso port `_forward_partial` replica a semântica mas recomputa o prefixo
inteiro [q0 + chunks 1..k] a cada passe — os chunks antigos são dead
weight: block-diagonal mascara toda a atenção fora do próprio nível.

## Achado 1 — chunk-local decode (VALIDADO, bit-idêntico)

Prova empírica (`profile_decode_chunk.py`, `artifacts/prof/decode_chunk.json`):
- logits último chunk, full-prefix vs chunk-local: **diff = 0.0 em k2/k5**;
  k8/k11 max 5,7e-6, **argmax 100% idêntico** (só reassociação de tiling fp)
- Timing B=8 @k10: 2143 → 412 ms/passe = **5,2× speedup medido**
- B=1: 254 → 97 ms = 2,6×
- Posições totais no decode intra (24 passes): 7996 → 1864 = **4,3×**

Implementação: novo `_forward_chunk(codes, k, cur, decided)` — monta só
[q0 + chunk k] com pos/lvl/rope indexados na posição GLOBAL (RoPE é
absoluta: q0↔chunk precisa da distância real). Máscara local: q0 self,
chunk atende tudo. RoPE q0 K/V é constante → cacheável por generate()
(ganho residual pequeno).

## Achado 2 — pipeline do eval: generate é 82% do wall

B=8 receita oficial (`profile_decode_pipeline.py`):
- generate 96,2s (24 passes × ~4s) — **o gargalo é um só**
- CE eval (2 fwd teacher-forced B=8): 19,6s — já ótimo (1 passe/nível)
- vqvae.decode + pick_tokens: **0,31s** — irrisório, não tocar

## Achado 3 — z_hat incremental (upstream parity)

`_chunks_from_codes` reconstrói z_hat acumulando todos os níveis a cada
passe O(K); upstream faz incremental. Fix: manter `z_hat_acc` pelo loop
de níveis — 1 embedding+interp por nível em vez de K.

## Achado 4 — decode adaptativo por confiança (literatura)

- **Fast-dLLM** (NVIDIA): KV cache block-wise + **decode adaptativo por
  threshold de confiança** (commits só acima do limiar) — até 27,6×.
  Aplicável direto: nosso intra decode commita janela fixa de 32; versão
  adaptativa commitaria posições confiantes → menos passes mantendo
  qualidade. Ressalva: threshold altera dinâmica de amostragem.
- **Ripple-Pivot Search** (arXiv 2608.11742): commits em posições
  pivô de entropia média propagam queda de incerteza — 4–10×.
- **Learn2PD**: filtro aprendido que prevê tokens corretos — 22–57×
  mas exige treinar um modelo auxiliar; overkill pro nosso orçamento.
- **WeDLM/SBD**: reformulações causais de decode — nosso modelo já é
  block-diagonal, estruturalmente mais limpo.

## Achado 5 — int8/engine C (adiado, mas fundamentado)

Zen2 tem AVX2+FMA sem VNNI: kernels int8 via `vpmaddubsw`/`vpmaddwd`
(ggml Q8_0, VelociQ8 Q8A64, qwen35.mojo W8A8 SignedDot ~12t/s num
i7-10610U) funcionam — ~1,5–2× no GEMV-bound B=1. Após chunk-local o
ganho relativo cai (dispatch já pequeno); manter como opção futura,
junto a torch.compile dos 11 grafos estáticos de decode.

## Estimativa consolidada

| alavanca | ganho estimado | risco |
|---|---|---|
| chunk-local decode | **4,3–5,2× medido** | nulo (bit-idêntico) |
| z_hat incremental | pequeno | nulo |
| intra_passes sweep (64) | ~1,3× em passes | tradeoff qualidade |
| decode adaptativo por confiança | 1,5–2,5× passes | protocolo novo |
| int8/engine C | 1,5–3× | port alto |
| combine chunk-local+adaptivo | **~6–12× total** | |

Prioridade óbvia: chunk-local + z_hat incremental — ganho medido,
risco zero, upstream parity.
