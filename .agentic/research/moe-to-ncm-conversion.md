# Research Brief — converter MoE pretrained para geração hierárquica em CPU

RAG local: `.agentic/research/rag/rag.py` (BM25, 591 chunks, 7 papers +
report + vendor docs). Consulta: `python .agentic/research/rag/rag.py "query"`.

## A analogia do usuário, decodificada matematicamente

O mapeamento renderização→LLM proposto é *exato*, não metafórico:

| Renderização | LLM equivalente |
|---|---|
| Render baixa-res + Lanczos/Mitchell upscale | geração coarse→fine multiescala (o NCM já é uma **pirâmide laplaciana** — Burt-Adelson 1983 aplicada a códigos; RVQ nível k = banda k do laplaciano) |
| Path tracing "de trás pra frente" (raios da câmera) | começar do rascunho barato e lançar "raios de correção" = draft→verify speculative / draft→edit refinement |
| Adaptive supersampling (mais amostras onde a variância é alta) | refinar só onde incerteza alta = níveis adaptativos por posição |
| Trocar resolução por custo | profundidade RVQ / subespaço ativado por token |

## Estado da arte cruzado (papers no RAG)

1. **Sub-MoE** (arXiv 2506.23266): merge de experts via SVD conjunta — U
   compartilhada entre experts + merge dos V específicos. 96% da performance
   com 25% de redução (Mixtral/Qwen MoE). **Cirurgia de pesos pura — zero treino.**
2. **MC-SMoE** (2310.01334): merge guiado por estatística do router + low-rank.
3. **Medusa-1** (2401.10774): cabeças paralelas sobre backbone CONGELADO —
   2,2× speedup lossless, só cabeças treinam. *É o NCM pragmático: paralelismo
   multi-token sem retreinar o tronco.*
4. **EAGLE**: drafting no espaço de features (penúltima camada).
5. **Adaptive/hierarchical softmax** (Morin-Bengio 2005; 1609.04309): árvore de
   vocabulário — 200× de speedup histórico prevendo classe→palavra.
6. **LCM (Meta, 2412.08821)**: conceitos = embeddings de sentença SONAR —
   geração em espaço latente contínuo (irmão conceitual do NCM).
7. **DELA/ME-DLM**: rascunho completo → edição local por controlador treinado,
   backbone congelado.
8. **RVQ on KV-cache** (2410.15704): EMA sem parâmetros aprendidos sobre
   ATIVAÇÕES de LLM pretrained — profundidade 8 recupera quase tudo. *Mesma
   receita do nosso MiniMSRQ — validada em LLM real.*

## A síntese original — "Pirâmide sobre tronco congelado"

Ninguém publicou a combinação: **RVQ hierárquico derivado do espaço de
embeddings do próprio LLM + cabeças de nível sobre backbone congelado**. Os
blocos existem todos; a composição é o espaço aberto.

### Componente A — codebook-pirâmide SEM treinar VQ-VAE (o atalho-chave)

Em vez de treinar um VQ-VAE (nosso gargalo passado), rodar **MiniMSRQ direto
sobre a matriz de embeddings do Qwen** (V×d). Cada token vira um caminho
L-níveis na pirâmide semântica: nível 0 = cluster grosseiro (≈"cor média da
região"), nível L = token exato. Reconstrução = soma dos códigos + lookup —
*lossless até o nível final*. É o equivalente a derivar a paleta de
quantização de uma imagem já renderizada, em vez de re-aprender óptica.

### Componente B — cabeças-de-nível (Medusa-1 no eixo de resolução)

Sobre o backbone Qwen congelado (0.6B ou 1.5B — o 30B fica como teacher):
L cabeças pequenas, cabeça_k prevê código_k do próximo token condicionada
pelos códigos <k. Treina SÓ as cabeças (~50-150M) — **CPU-feasível em dias**.
Geração: 1 passada do tronco emite a pirâmide inteira do token (coarse→fine
num único shot). Opcional: cabeças estilo Medusa para 2-3 posições futuras =
pirâmide no tempo E na resolução.

### Componente C — cirurgia MoE→denso para o teacher/barato (Sub-MoE)

Aplicar Sub-MoE/SVD-merge nos 128 experts do Qwen3-30B-A3B → ~16-32 experts
efetivos (ou denso aproximado). Ganho duplo: (1) teacher mais rápido para
gerar corpus de destilação em CPU; (2) testa se a "resolução" do MoE pode
cair sem perda relevante — medida direta da hipótese adaptive-sampling.

### Componente D — o knob "resolução" em inferência (a novidade publicável)

Inferência adaptativa: nível k decide refinar para k+1 por entropia.
Tokens previsíveis (pontuação, boilerplate de código) param no nível 2-3 —
palavras de conteúdo vão até o fim. = **adaptive supersampling exato**, com
custo por token variando dinamicamente. Nenhum paper no RAG faz isso sobre
cabeças de pirâmide.

## O que cabe na ThinkPad 32GB

| Peça | RAM | CPU/dia |
|---|---|---|
| MiniMSRQ sobre embeddings Qwen-0.6B/1.5B | <2GB | minutos-horas |
| Sub-MoE merge (SVD nos experts, offline) | ~20GB picos | horas |
| Treino cabeças (backbone 0.6B congelado) | ~4-6GB | dias — dentro do regime |
| Feature-collection do teacher 30B (Q4) | ~20GB | lento — overnight ok |

## Riscos honestos

- A pirâmide pode colapsar como nosso VQ-VAE colapsou — mitigação conhecida
  (MiniMSRQ EMA-bounded já resolveu isso uma vez).
- Cabeças-de-nível sobre tronco congelado podem ter teto baixo — o paper da
  Medusa mostra que cabeças sobre backbone congelado funcionam (2,2×), mas
  para códigos RVQ em vez de tokens é território novo.
- O ganho pode ser marginal vs speculative simples — por isso Phase A mede
  fidelidade de quantização ANTES de qualquer treino.

## Reuse check

MiniMSRQ (nosso, já provado), trainer existente, Qwen-0.6B GGUF local,
Qwen3-Coder-30B-A3B local, llama.cpp (teacher+speculative baseline).
Só dependência nova possível: nenhuma — tudo numpy/torch local.
