# Pesquisa: por que o NCM oficial funciona e o nosso falhou

Fonte primária: **"NCM: Next-Concept Modeling: Fast, Parallel Coarse-to-Fine
Text Generation"** (arXiv:2610.14162, Stanford/HazyResearch, out/2026) —
é O paper no qual este projeto se inspira. Código vendored em
`vendor/ncm-upstream` é o repo deles (github.com/HazyResearch/ncm).
Checkpoints publicados: huggingface.co/hazyresearch/ncm-checkpoints
(TinyStories incluso — baixado em `artifacts/ncm_official/`).

## Verificação empírica do modelo oficial na nossa CPU (container lcm-infer)

- Amostras incondicionais: **estórias coerentes de verdade**
  ("Once upon a time, there was a little girl named Lily...")
- Velocidade: 4,4-4,5s / 8 amostras ≈ **0,55s/amostra** (nosso AR 0,62s)
  — em CPU a vantagem some: o speedup ×10-30 deles é medido em GPU,
  onde os passes paralelos escalam; em CPU o custo por passo domina.
- Teacher-forced CE por nível (4 textos, V=4096, unif=8,32):
  l1=7,92 l4=5,60 l7=3,98 **l11=2,75** — os códigos finos deles
  SÃO previsíveis: aos 11 níveis ẑ já quase reconstrói z, o resíduo
  final é pequeno e regular.
- Tokenizer deles nos nossos textos: reconstrução perfeita na parte
  visível (0,43 acc = artefato de padding).

## As diferenças mecânicas (receita oficial vs nossa)

| Componente | Oficial | Nosso |
|---|---|---|
| Tokenizer | 36,4M params, 6 layers, emb 384, groupnorm, qk_norm, 40k steps | 6,35M, emb 192, ~3k steps |
| Nested dropout | `fine_drop.prob=0.5` (reconstrução avaliada em corte aleatório) | ausente |
| quant_resi | ativo, ratio 0.5 | inert |
| learned_l1 | sampling+cascade base 4 | ausente |
| Hierarquia | **12 níveis** [1,8,16,24,32,40,52,64,76,92,108,128] | 6 níveis [1,8,16,32,64,128] |
| Codebook | **uniforme 4096 em TODOS os níveis** | 2048 uniforme / tapered |
| Gerador | 88,6M, 12 layers, 768d, tied-codebook heads | 62M, 4 layers, 896d |
| Corrupção de códigos | ativa (média 0,5, linear decrescente por nível) | existia mas **inerte** |
| Treino gerador | 50k steps @ batch 2048 (~13B posições) | 3k @ batch 16 (~6M) |
| Nível 0 | amostrado de unigram | predito |
| Atenção | **block-diagonal** (vence block-causal 15,4 vs 19,6 PPL) | block-diagonal ✓ |

## Releitura do nosso resultado negativo

Nossa conclusão "códigos residuais finos são imprevisíveis" era correta
**para a nossa representação/treino** — não é uma lei da arquitetura:
- hierarquia de 6 níveis deixa salto grande demais para o último nível
  (64→128 posições com todo o detalhe restante);
- tokenizer fraco + sem nested dropout → informação não distribuída;
- sem corrupção → exposure bias nos níveis grossos também;
- ~2000× menos tokens de treino do gerador.

O diagnóstico GT-fronteira segue válido para nossos códigos (I≈0 nos
finos). O paper prova que, com a receita certa, residual-VQ de texto
funciona — a lacuna era de implementação/escala, não de princípio.

## Otimizações ordenadas por custo/valor

1. **[mais barato, ~6-8h] Tokenizer oficial + nosso gerador + corrupção**:
   tokenizer.pt já baixado e provado (códigos previsíveis, l11 CE 2,75);
   encode corpus_ts (~30min) → adicionar corrupção em `loss()` (~30min)
   → treinar LevelGenerator 3-6k steps. Nosso gerador já é compatível
   com a API do vq deles (embeddings/_up_to_full/fallback area).
2. **[~2h] Adicionar fine_drop + corrupção ao MiniMSRQ e retreinar nosso
   VQ** — testa se os MECANISMOS (não só escala) bastam na nossa escala.
3. **[zero custo] Usar o stack oficial como referência NCM** nos
   relatórios — já funciona na CPU (0,55s/amostra).
4. **[~dias, caro] Replicação completa**: tokenizer deles 40k steps +
   gerador 12-layer 10-50k steps @ batch 1024+ — inviável em CPU
   convencional no horizonte; só se GPU aparecer.

## Referências consultadas

- NCM paper: arXiv:2610.14162 (+ repo oficial, HF checkpoints)
- VAR (arXiv:2404.02905): next-scale em imagem funciona porque imagem é
  localmente suave — resíduos finos previsíveis do grosso
- MagViT-2/LFQ (arXiv:2310.05737): quantização binária + entropia
  (confiança por amostra + diversidade de batch)
- FSQ (arXiv:2309.15505): quantização escalar fina, sem codebook
- SpeechTokenizer (arXiv:2308.16692) / Mimi / XY-Tokenizer: destilação
  semântica no 1º quantizador RVQ — grosso=semântico, fino=detalhe;
  mesmo padrão do nested dropout do NCM
- LLaDA (NeurIPS 2025): difusão mascarada como alternativa de geração
  paralela para texto — SOTA não-AR; lenta em poucos steps
- ResGen (PMLR v267): predição conjunta de embeddings RVQ
