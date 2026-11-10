# Avaliação de arquitetura: l0 externo (unigram) na geração NCM

## Estado

- `LevelGenerator.generate()` é stateless por chamada; `self.last_passes`
  é o único estado escrito (contador de passes) — manter semântica:
  l0 externo ainda conta como +1 pass.
- Novo dado: distribuição unigram de l0 (vetor V0=4096). Fonte:
  `codes_official/train.npy[:,0]` → `np.bincount`. Persistir como
  `.npy`/`inference.level0_unigram` — espelha convenção upstream
  (`generator.pt:inference.level0_unigram`).

## Fluxo

```
train.npy[:,0] ──bincount──▶ l0_unigram.pt
                                     │
generate(l0=multinomial(l0_unigram)) ▼──► trunk P(l1..l11 | l0) ──► vqvae.decode
```

Assinatura alvo: `generate(batch_size, ..., l0=None)` — `l0 is None` →
comportamento atual (modelo prediz, preserva backward-compat e o
Experimento C).

## Computação

- `torch.multinomial` sobre 4096 — custo desprezível vs 12 passes do trunk.
- `codes_official/train.npy` é read-only (177k×641 int16/int32) — apenas
  coluna 0 lida; não mutar.

## Falhas e fronteiras de confiança

| Falha | Mitigação |
|---|---|
| l0 externo fora do range [0,V0) | assert shape/range no generate |
| Unigram de corpus divergente do ckpt do gerador | unigram vem do MESMO corpus usado no treino v2 — verificar soma=177175 e nunca-vistos=0 |
| Comparabilidade com bench_v2 | mesma seed/temp/topk; seed controla multinomial l0 também |
| Arquivo `.npy` carregado inseguro | `np.load(..., allow_pickle=False)` |

## Trade-offs

- **Unigram empírica vs `inference.level0_unigram` do ckpt oficial**: usar
  o cálculo local (mesma distribuição, proveniência limpa p/ paper).
  Comparar os dois como sanity check.
- **`--l0 active/uniform/smooth`**: implementar só `unigram` + `smooth:eps`
  — `active`/`uniform` são ablações triviais depois.
- **Não portar `pick_tokens` anti-repetição agora**: config oficial
  TinyStories não o usa — stutter residual após l0 correto vira
  Experimento B isolado (variável controlada).

## Critérios de validação

- l0 amostrado ∈ códigos vistos no corpus (uniq ≤2706)
- Amostras com l0-unigram mais coerentes que bench_v2 (avaliação por
  repetição: taxa de bigramas duplicados "X X" + leitura qualitativa)
- bench JSON registra `l0_mode` p/ rastreabilidade
