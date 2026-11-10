# Pesquisa: l0 unigram — a correção sem retreino para coerência global

**Contexto**: RUN-023. Diagnóstico pós-v2 (RUN-019/020): amostras v2 têm
registro TinyStories mas gagueira massiva + arco ausente. l0 held CE ≈ 8,2 ≈
uniforme (V=4096 → ln=8,32). Pergunta: escala ou defeito arquitetural?

## Achado — o oficial nunca prediz l0

`vendor/ncm-upstream/scripts/eval_shard.py` + `artifacts/ncm_official/config.yaml`:

- `predict_level0: False` no modelo oficial
- `generation.level0_unigram: generator.pt:inference.level0_unigram` —
  a distribuição empírica de l0 do corpus de treino é **salva dentro do
  checkpoint** e amostrada por `torch.multinomial(L0_P, B)` (eval_shard.py:156)
- Modos `--l0`: `unigram` (default) | `active` (uniforme sobre usados) |
  `uniform` | `smooth:<eps>` (mistura unigram+uniforme)
- Amostragem de códigos TinyStories: `temperature=1.0, top_k=0, top_p=1.0`
  — **multinomial plena**, não greedy
- Anti-repetição existe no arsenal (`frequency_penalty`,
  `frequency_allowance`, `adjacency_penalty` em `pick_tokens`) mas **não é
  usada no config TinyStories** — a coerência deles é mérito do modelo, não
  de penalidade de decode

## Por que isso explica (parte da) nossa falha

| | Oficial | Nosso (v2) |
|---|---|---|
| l0 no treino | GT l0 alimenta o trunk | GT l0 alimenta o trunk ✓ |
| l0 na inferência | ~unigram empírica (in-distribution) | ~modelo (≈uniforme, lixo) |

Consequência: o trunk aprendeu P(níveis 1..11 | l0 real). Na nossa
inferência, l0 sai de uma distribuição ≈uniforme sobre 4096 códigos —
incluindo os ~1400 códigos **nunca vistos** no corpus (uniq l0 = 2706).
Todo o condicionamento nasce fora de distribuição → deriva + degeneração.
Correção: amostrar l0 da unigram empírica. **Zero retreino.**

## Experimento A (barato, ~1h): l0 unigram

1. Unigram: `inference.level0_unigram` de `artifacts/ncm_official/generator.pt`
   OU `np.bincount(codes_official/train.npy[:,0])` — preferir o segundo
   (proveniência local); conferir que batem.
2. Patch `generate()`: aceitar `l0` externo (tensor B×L0) — espelha
   `net.generate(l0, x, ...)` do upstream.
3. Re-bench das 8 amostras, mesma seed/temp do bench_v2 → comparar.
4. Sweep rápido: temp 1.0 top_k 0 (receita oficial) vs temp 0.8 top_k 40.

## Experimento B (se gagueira persistir): anti-repetição oficial

Portar `pick_tokens` (freq/allowance/adjacency) como pós-decode opcional —
medir quanto do stutter é degeneração de amostragem vs qualidade de modelo.
Também testar `--l0 active` e `smooth:0.05`.

## Experimento C (overnight, opcional): mais steps

Resume de `gen_ncm_v2/final.pt` por +10-15k steps — testa se l0 modelo
aprende com orçamento (pergunta científica que sobra); independente de A.

## Literatura adjacent (a pesquisar a cada descoberta)

- Jukebox (Dhariwal 2020): VQ-VAE hierárquico, prior separado p/ nível topo
- RQ-Transformer (Lee 2022): primeiro código de imagem também é o difícil
- MaskGIT/VAR: amostragem de nível grosso com CFG (`cfg_scale` já suportado
  no upstream — `predict_level0`/`query_level0` existem como opções)
- LFQ/MagViT-2: regularizadores de entropia p/ codebook
- Welleck et al. 2019: degeneração de amostragem neural (stutter)

## Reuse check

- `.agentic/research/ncm-official-recipe-analysis.md` já flaggava
  "Nível 0: amostrado de unigram (oficial) vs predito (nosso)" — confirmado
  agora com a mecânica exata e os defaults de sampling
- `artifacts/codes_official/train.npy` (177k×641) — l0 = coluna 0
- `artifacts/gen_ncm_v2/final.pt` — checkpoint alvo do Experimento A
- `bench_v2.json` — baseline de comparação (8 amostras, temp 0.8, topk 40)
