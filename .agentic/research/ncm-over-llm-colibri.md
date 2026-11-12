# Research note — treinar NCM "sobre" o LLM do colibri?

Demanda (RUN-033): avaliar se o treinamento pode migrar para o modelo
testado no colibri (Qwen3-Coder-Next 80B IQ1_M / 30B IQ2_M-topk6).

## T1 — Os modelos do colibri são treináveis localmente? NÃO.

Evidência:

- **Formato**: GGUF é serialização de *inferência* (llama.cpp).
  Pesos quantizados (IQ1_M ~1.7bpw) não carregam gradientes nem
  suportam autograd — não existe caminho de treino nesse formato.
- **Memória**: máquina tem 30GB. Pesos bf16: 30B ≈ 60GB, 80B ≈ 160GB —
  os pesos *desquantizados* já não cabem, antes de contar otimizador
  (AdamW adiciona ~8B/param → +240–640GB).
- **Computação**: nosso gerador NCM de 64M params leva ~16s/step.
  Um LLM de 3B+ seria >50× os FLOPs por token — sem sentido em CPU.
- **Contenção já medida no colibri**: o 80B precisa ~24GB livres e com
  o treino comendo 16–19GB ele cai para ~0.5 t/s — nem inferência
  simultânea cabe direito.

Conclusão T1: **impossível treinar/fine-tunar esses modelos nesta
máquina.** Isso é fato de física de memória, não de engenharia.

## T2 — O que "usar o modelo do colibri" PODE significar (viável)

| alternativa | o que é | custo | valor |
|---|---|---|---|
| **A. Teacher corpus** | LLM gera corpus de código; retreinamos tokenizer VQ + gerador NCM nesse corpus | corpus: horas–dias de inferência (30B ~20-40 t/s com máquina livre); retreino ~12-24h | **alto — é o caminho real ao "NCM para código"**, objetivo final do estudo |
| B. VQ sobre hidden states do LLM | tokenizer VQ-VAE treinado no espaço de representação do LLM | extração de hidden states via llama.cpp é limitada; engenharia nova | incerto/risco alto |
| C. LLM como l0 híbrido | LLM gera o código global (l0), NCM refina os níveis finos | inferência apenas | especulativo, muda a tese |
| D. Continuar braços TinyStories (l0) | status quo | ~12h/braço | incremental |

A opção A é a única que "migra o treinamento" de fato: o LLM vira
**fábrica de dados** (teacher), não o aluno. O pipeline NCM inteiro
(tokenizer VQ residual + gerador hierárquico) se retreina do zero
sobre corpus de código — exatamente a pergunta que motivou o estudo
(representações hierárquicas para código).

## T3 — Recomendação

1. **NÃO interromper o v5-ext** (~1.000 passos restantes, ~4h): é o
   braço científico em voo; pará-lo perde a trajetória.
2. Quando a máquina liberar (~16GB devolvidos), decidir:
   - **A**: gerar corpus de código com o 30B IQ2_M (cabe nos ~24GB
     livres; o 80B fica marginal mas possível a ~30-45 t/s) → braço
     "NCM-code" = primeira evidência real sobre código.
   - **D**: braço l0/coerência em TinyStories (fronteira científica
     atual do paper).
3. Decisão A vs D é do usuário — ambas são legítimas; A conecta o
   estudo ao objetivo final, D fecha a pergunta aberta do paper.

## Limites honestos

- Nada disso faz o 80B/30B *aprender* — são pesos congelados.
- Um corpus gerado por LLM herda os vieses do teacher (o estudo já
  documenta mode-collapse por refinement — mesmo mecanismo em escala
  de dados).
- "NCM para código" é um braço de ~2-3 dias de máquina dedicada
  (corpus + tokenizer + gerador), não uma continuação trivial.
