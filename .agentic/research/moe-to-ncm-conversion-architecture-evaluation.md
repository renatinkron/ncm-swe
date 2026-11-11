# Architecture Evaluation — "Pirâmide sobre tronco congelado" (MoE→NCM)

## State

- Backbone congelado (Qwen-0.6B/1.5B): read-only pesos — estado imutável por
  design; toda mutação vive em cabeças + codebook → falhas não tocam o teacher.
- Codebook-pirâmide: artefato determinístico (kmeans/EMA sobre matriz fixa).
- 30B teacher: só leitura/inferência — nunca treinado (inviável por RAM mesmo).

## Flow

embeddings Qwen → MiniMSRQ (L níveis) → pyramid-codes por token →
features do backbone (forward, congelado) → treino cabeças L →
inferência adaptativa (early-exit por entropia) → decode pirâmide→token.

## Computation

- Phase A (codebook): numpy/torch, minutos — valida fidelidade antes de gastar.
- Phase B (cabeças): forward congelado 0.6B + backward só nas cabeças —
  ~5-10s/step CPU estimado → 2000 passos ≈ horas-dias. Viável.
- Phase C (Sub-MoE): SVD 128×experts — linear algebra, horas.
- Inferência: 1 passada → pirâmide; adaptativa reduz código médio por token.

## Failure analysis

- **Codebook degenerado**: detectável em Phase A (normas, códigos vivos) —
  mesmo diagnóstico do colapso anterior; MiniMSRQ já é a correção.
- **Cabeças não aprendem**: baseline = loss < uniforme por nível + monotonia
  coarse-first (a assinatura que já observamos nos 471M).
- **Custo de feature-collection**: se forward do tronco for gargalo, cache
  hidden states uma vez por sequência (imutável) e retreine cabeças em cima.
- **Early-exit mata qualidade**: medir per-level ablation no bench — o knob
  de resolução só entra em produção com curva qualidade×nível.

## Trust boundaries

- Nenhum componente executa código gerado. Pesos Qwen locais verificados.
- RAG é read-only, corpus = papers arXiv oficiais.

## Verdict

A composição é coerente e em grande parte *conservadora*: cada peça tem
evidência independente (Medusa-1, Sub-MoE, RVQ-EMA, hierarchical softmax) —
o novo é a união. Phase A (codebook→fidelity) é o gate barato que decide
se vale investir dias no resto. Proceder.
