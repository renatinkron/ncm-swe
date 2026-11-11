# Architecture Evaluation — pipeline de treino curado NCM

## State

- Corpus novo: fatia curada (download HF) + slice sintético (30B
  data factory). Novo tokenizer BPE → VOCABULÁRIO e embedding space
  mudam → codebook RVQ, heads, stream de códigos todos REBUILDADOS
  para o novo domínio (artefatos antigos incompatíveis por design).
- Modelo NCM nativo (gen_ncm existente) + decoder conceito→token
  co-treinado — peça nova de estado (decoder junto ao gerador).

## Flow

download/gera corpus → tokenizer (sentencepiece) → stream tokens →
codebook RVQ sobre embeddings do PRÓPRIO modelo em treino OU
codebook separado? — decisão de design: RVQ sobre embeddings
aprendidos muda durante o treino (moving target). Opções: (a)
congelar codebook após warmup; (b) RVQ online (dead-code restart).
Decisão adiada p/ spec — ambas têm precedente (RQ-VAE treina
codebook junto com commitment loss).

## Computation

- Tokenizer: minutos. Codebook+stream: ~1h (escala do Qwen-0.5B
  foi minutos; modelo novo ~igual). Treino NCM 30-100M + decoder:
  estimado ~24-72h CPU p/ 50-200M tokens × ≤4 épocas — a parte
  pesada, compatível com runs anteriores (471M×2k steps=14h).
- 30B data factory: 1-5M tok/dia — slice sintético limitado a
  ~2-10M tok (5-10% do corpus), suficiente p/ efeito Cosmopedia.

## Failure

- Vocabulário próprio quebra transferência p/ modelos maiores —
  aceitável (demo científica).
- Corpus sintético pequeno demais p/ efeito → mitigação: fatia
  curada pronta domina o corpus; sintético é tempero.
- Moving codebook → mitigation (a) freeze após warmup.

## Trust boundary

Datasets HF públicos (revisar licença: TinyStories CDLA, MiniPile
MIT/Pile); 30B local GGUF já no disco — sem código de terceiros
executado no host além de sentencepiece/tokenizers (libs padrão).

## Validation criteria

- Acceptance nativo > 0 (spec interno do NCM, não vs AR externo)
- Speedup end-to-end NCM vs AR casado mesmo corpus/params
- Coerência: parse de história/código gerado (eval TinyStories-style)
