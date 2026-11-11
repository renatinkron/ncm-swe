# Treino eficiente com dados curados — receitas frontier (peer-reviewed)

## Context

Pergunta do usuário: os modelos de fronteira obtêm resultado com
**menos dados de melhor qualidade** — quais técnicas aplicam ao nosso
NCM de escopo fechado em CPU? Resposta: existe uma receita
documentada e replicável em 5 pilares.

## Constraints

- CPU 4750U: treino realista = 30-150M params, corpus 100M-1B tokens,
  épocas ≤4. Geração de dados sintéticos só via modelo LOCAL
  (30B-A3B GGUF parado no disco, ~1-5M tok/dia).

## Pillars (fontes)

1. **Textbook-quality >> quantity — Phi-1** (arXiv 2306.11644):
   1.3B treinado em 6B tokens filtrados "nível-textbook" + 1B
   sintéticos GPT-3.5 (~8 épocas, 50B vistos) → 50,6% HumanEval;
   phi-1-small 350M → 45%. Filtragem por classificador +
   GPT-3.5 p/ avaliar "educational value". **Prova: qualidade
   quebra scaling law.**

2. **Escopo fechado + vocabulário pequeno — TinyStories** (arXiv
   2305.07759): corpus sintético GPT-3.5/4 restrito a ~1.500
   palavras (prompt template: verbo+substantivo+adjetivo + feature
   narrativa → diversidade forçada); modelos 1-33M geram
   parágrafos coerentes. **O paradigma exato da nossa demo.**

3. **Receita completa documentada — SmolLM/SmolLM2** (arXiv
   2502.02737 + HF blog): (a) classificador educacional (Llama-70B
   anota, fastText/scorer filtra — FineWeb-Edu removeu 92%);
   (b) Cosmopedia-v2: 28B tok sintéticos Mixtral, diversidade por
   seed × audience × format; (c) modelos ≤360M: single-stage só
   com dados bons; WSD schedule (2k warmup, pico 5e-4, decay
   últimos 10%), AdamW β=(0.9,0.95).

4. **Épocas ≤4 são quase grátis — Muennighoff** (NeurIPS'23/JMLR'25
   datablations, 400 runs): repetir dados até 4× ≈ dados únicos;
   além disso decai a zero. Corpus curado pequeno pode ser
   multi-época sem custo de qualidade.

5. **Destilação sequence-level — MiniLLM/MiniPLM** (ICLR'24
   2306.08543; arXiv 2410.17215): professor gera texto (SeqKD),
   aluno treina nele — sem compartilhar tokenizer, sem logits.
   MiniPLM refina a distribuição do corpus ("difference sampling":
   amostrar onde professor ≫ aluno-pequeno). **Nosso 30B-A3B vira
   data factory**: SeqKD local, offline, já no disco.

6. **Curadoria pronta barata — MiniPile** (arXiv 2304.08442): 6GB/
   1M docs cluster-filtrados do Pile → BERT/T5 a 1,9-2,5% do
   full-data. E `iNeil77/pseudo-mini-pile`: receita de filtros
   baratos (perplexity KenLM, MinHash dedup, n-gram rep).

## Options

A. **Corpus pronto** (TinyStories/Python-Edu/MiniPile-slice):
   zero custo de professor, começa hoje. Risco: domínio menos
   alinhado a código-NCM.
B. **Corpus sintético local** (30B como teacher, Cosmopedia-style):
   alinhado ao domínio código; ~1-5M tok/dia → slice de 2-10M tok
   em dias, misturável com (A).
C. **Híbrido (recomendado)**: download de fatia curada (esqueleto
   do corpus) + slice sintético do 30B (temas/estilo controlados,
   tipo exercícios tipo-Phi + texto didático) + tokenizer próprio
   BPE 4-16k sobre o corpus final.

## Recommendation

Receita híbrida TinyStories×SmolLM adaptada a CPU:
- Corpus-alvo ~50-200M tokens: 60-80% curado pronto + 20-40%
  sintético do 30B (axes: tópico × audiência × formato,
  restrição de vocabulário TinyStories p/ alinhar com o tokenizer)
- Tokenizer BPE próprio (4-16k) — a última milha RVQ vira tratável
- Modelo NCM nativo 30-100M (gen_ncm já existe) com decoder
  conceito→token **co-treinado** — a peça que sempre faltou
- ≤4 épocas (Muennighoff), WSD schedule, AdamW β2=0,95
- Eval: acceptance nativo + speedup end-to-end (a métrica que
  nunca foi medida por inteiro)

## Open Questions

- Domínio final: Python simples vs estórias simples (usuário decide)
- Vocabulário: 4k (seguro) vs 16k (ambicioso)
- Throughput real do 30B como data factory (medir antes de prometer
  slice sintético grande)
