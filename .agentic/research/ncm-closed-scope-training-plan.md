# Spec v1 — NCM nativo em escopo fechado (treino eficiente)

Fontes: `.agentic/research/efficient-curated-training-peer-reviewed.md`
(Phi-1, SmolLM2, TinyStories, Muennighoff, MiniLLM, MiniPile).

## 0. Medidas empíricas base (RUN-013)

- 30B-A3B data factory: **15,7 tok/s gen** (prompt 38,9 t/s) →
  ~1,35M tok/dia de texto didático de boa qualidade (amostra
  verificada: seção de textbook Python correta e limpa).
- Datasets confirmados vivos + licenças OK.

## 1. Corpus-alvo: ~60-150M tokens

| Fatia | Fonte | Share | Custo |
|---|---|---|---|
| Curado pronto | TinyStoriesV2-GPT4 (2,2GB ~500M tok) OU python-edu slice (~50-100M tok) | 70-90% | ~10-30min download |
| Sintético local | 30B-A3B Cosmopedia-style (axes: tópico × audiência × formato, restrição vocab) | 10-30% | 1-7 dias bg |

Domínio default: **estórias TinyStories** (vocabulário naturalmente
fechado ~1,5k palavras, dataset pronto, GPT-4-only V2) — é o
paradigma mais limpo p/ provar o pipeline. Alternativa: python-edu
(domínio código, north-star do projeto) com vocabulário um pouco
maior. Usuário escolhe; ambos cabem na receita.

## 2. Tokenizer próprio

sentencepiece BPE no corpus final: vocab **4k** (conservador) ou
**16k** (ambicioso). Minutos de CPU. Efeito: vocabulário 4-16k vs
152k → codebook RVQ e última milha ficam tratáveis (o problema
medido em todas as RUNs 007-012).

## 3. Modelo NCM nativo + decoder co-treinado

- Backbone transformer 30-100M params (gen_ncm existente), emb dim
  ~512-896, tied embeddings.
- **Codebook RVQ** sobre os embeddings (rebuild p/ o novo vocab —
  artefatos Qwen incompatíveis): 4-6 níveis, esperado top-1 de
  reconstrução muito acima dos 32% medidos em 152k.
- **Decoder conceito→token co-treinado**: cabeça que recebe
  (hidden + vetores de código previstos) → logits do token —
  treinada JUNTO (a peça que sempre faltou: nas runs anteriores
  era parafusada depois, congelada, desalinhada). Equivalente às
  nossas LevelHeads+TokenHead mas nativas e co-otimizadas.

## 4. Receita de treino (frontier, adaptada a CPU)

- AdamW β=(0.9,0.95), WSD schedule: warmup ~1-2k steps, pico
  ~5e-4, decay nos últimos 10% (receita SmolLM2, arXiv 2502.02737)
- **≤4 épocas** sobre o corpus único (Muennighoff NeurIPS'23:
  repetição até 4× ≈ dados únicos)
- batch ~64×seq256 ou equivalente (~16k tok/step)
- Checkpointing por step + eval periódico (loss + acceptance
  nativo + amostras de texto)

## 5. Evals (o que nunca foi medido inteiro)

1. **Acceptance nativo** — draft piramidal do PRÓPRIO modelo vs
   seu verify AR (não vs Qwen externo): o acceptance real do NCM.
2. **Speedup end-to-end** — NCM vs AR casado (mesmo backbone/params,
   mesmo corpus) — reproduz o ×2,27 do 471M com a última milha
   funcionando.
3. **Coerência** — parse/fluência das amostras (estilo TinyStories:
   gramática, consistência; para código: % sintaticamente válida).

## 6. Estimativas honestas de tempo

- Tokenizer+codebook+stream: ~1-3h
- Treino 30-100M × 60-150M tok × ≤4 ép: ~1-4 dias CPU
  (referência: 471M×2k steps=14h em dados piores)
- TOTAL primeira demo end-to-end: ~2-5 dias, com etapas em
  background e checkpoints (não bloqueia a máquina o tempo todo)

## 7. Riscos e mitigações

- Moving codebook durante treino → congelar após warmup ou
  commitment loss (RQ-VAE precedent).
- Slice sintético insuficiente → curado domina; sintético é tempero.
- Domínio errado → TinyStories é a aposta segura documentada.
