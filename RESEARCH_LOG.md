# RESEARCH_LOG — trilha de pesquisa (breadcrumb reproduzível)

Documento vivo. Objetivo: qualquer pessoa (ou agente) conseguir entender onde
estamos, por que cada decisão foi tomada, e reproduzir do zero.

Última atualização: pós RUN-047 fechado + RUN-048 formal (painel NCM, agente
paralelo). Linha intent/v3 aqui documentada foi probe de sessão — ver §4.

---

## 1. A intenção (o que o usuário quer de verdade)

Construir, em CPU comum (~32GB RAM, sem GPU dedicada, laptop Linux), um modelo
de código pequeno que **converse com o usuário em linguagem natural e gere
código funcional** — e que possa absorver conhecimento transferido de um
professor maior (eu, o agente, e/ou um Qwen local) sem treinar do zero.

Intenções capturadas ao longo da conversa (ordem cronológica):

1. Modelo pequeno rápido para código — existe e funciona (55M BPE).
2. "Quero que você gere GBs do seu conhecimento no formato que o modelo
   estoca, tipo um GGUF do seu MoE" — **refinado**: o usuário intuiu um
   artefato de conhecimento em massa; a forma correta não é pesos, é
   **memória externa indexada** + canal de instrução.
3. "Reproduzir padrões, não a complexidade" — a Mona Lisa: não queremos o
   córtex do Da Vinci, queremos as pinceladas. Nome técnico: **distillation
   comportamental**.
4. "Logits do Qwen em paralelo?" — explorou o canal mais denso de
   transferência; medido inviável entre vocabulários diferentes (§6).
5. Fast feedback sempre: probes curtos (~25min) antes de qualquer run longo.
6. Registro contínuo: este arquivo + `SESSION_HANDOFF.md` + REPL para
   inspeção manual pelo próprio usuário.

## 2. O que o sistema é hoje (estado)

```
Gerador:        transformer decoder 55M params, ctx=256, byte-BPE 8192 vocab
Checkpoint:     gen_bpe_currv3.pt (melhor) / gen_bpe_curr.pt (v2)
Treino:         CPU-only, podman rootless (lcm-infer), --network none
Dados base:     ~26k func_units reais (leg-1) + legs subsequentes
Teacher:        eu (agente) — units manuscritos elite + synth generator
Evals:          heldout CE/top1 · argmax/sampled parse·real·filler ·
                deep eval por prefixo · eval de intenção (novo, v3)
Inspeção:       ./repl.sh <ckpt> — REPL interativo argmax+sampled
```

## 3. Resultados medidos — a matriz completa

| modelo | CE real | BOS argmax | BOS sampled | 'def ' topk24 | 'def ' argmax | custo |
|---|---|---|---|---|---|---|
| leg-1 (corpus) | 0.9162 | 0/32 | 16/32 | 26/32 | 0/8 | baseline |
| leg-2 (+6h) | 0.9345 | 0/32 | 18/32 | — | 0/8 | saturado |
| teacher FT puro | 1.3543 | 32/32 | 32/32 | 21/32 | 8/8 | forgetting +0.44 |
| mix 15% | 0.9392 | 32/32 | 10/32 | 18/32 | 0/8 | perturba sampled |
| **curr v2** | 1.044 | 32/32 | 32/32 | **32/32** | 8/8 | melhor geral |
| **curr v3** | — | 32/32 | 32/32 | — | — | +canal intenção |

Prefixos argmax do vencedor (v2): `def ` 8/8 ✓ · `async def ` 8/8 ✓ ·
`def _` 8/8 · `class ` 0/8 · `def test_` 0/8 — buracos remanescentes
conhecidos e documentados.

## 4. As impossibilidades encontradas — e como foram resolvidas

| Impossibilidade | Por quê | Solução adotada |
|---|---|---|
| "Gerar o GGUF/pesos direto" | Pesos são resultado de otimização, não texto autoral; sem backprop não há como calcular os 55M floats | Eu escrevo o **livro** (dados/currículo); o treino constrói o "cérebro". GGUF é container de floats, não conhecimento |
| "Transferir meus logits" | Eu não tenho acesso introspectivo aos meus próprios tensores/ativações — sei *como* sou computado, não *o que* meus números são | Canal texto: eu emito artefatos (units, curricula); retenção é medida por cross-CE |
| "Logits do Qwen como professor" | Vocabulários diferentes (8192 vs ~150k) e fronteiras de token desalinhadas — logit-KD clássico exige mesmo alfabeto | Sequence-level KD (texto é o canal) + logprob como metadado opcional |
| Logprob como filtro de confiança | Medido: token_logprob ≈ 0.000 em tudo — sinal plano, não discrimina | Descartado como critério; filtro passa a ser parse + heurística + revisão minha |
| "Mais épocas no corpus" | leg-2: +6h → heldout piorou (+0.018), só memorizou | Corpus saturado confirmado; mudou o eixo pra teacher-data |
| Teacher FT puro | +0.44 CE de forgetting — dose total substitui a distribuição | Curriculum: teacher como **fase final** sobre base convergida (LR 1e-4, ~320 steps) |
| Mix raso 15% | Pior dos dois mundos: sampled caiu sem transferir hábitos | Dose tem resposta não-monotônica — ou final-curriculum ou nada |

## 5. Linha do tempo de direcionamentos

```
RUN-043/044: VQ-641 provado não-ser o problema → canal VQ ~25% teto
RUN-045:     direct-BPE + func_units → 18/32 (direção validada)
RUN-046:     ctx 256 + BOS in-dist → 26/32 (recorde corpus)
             ⚠ PIVOT SILENCIOSO: "matar VQ, CPU pro BPE" trocou a TESIS
             (arquitetura disruptiva) pela META (gerador que funciona).
             Decisão certa pra capacidade, mas a tese NCM não morreu —
             foi só abandonada sem declaração. Registro honesto aqui.
RUN-047:     teacher probe → curriculum B vencedor → FECHADO
             achado central: teacher-data é shift de distribuição;
             a posição da dose importa mais que a dose
RUN-048:     (formal, agente paralelo) painel renderizador NCM +
             dump LoRA + testes — a visão do usuário ganhou interface
probes v3:   canal intenção→código confirmado 61%; Qwen logits inviável;
             baseline 0.6B pronto = 83% → capacidade vem de pretraining
```

## 6. O experimento Qwen (medido, não especulado)

Setup: `llama-server` local, `Qwen3-1.7B-Q8_0` e `Qwen3-Coder-30B-A3B
IQ2_M-topk4`, porta 8085, ctx 4096, 8 threads.

| probe | parse | obs |
|---|---|---|
| 1.7B, 20 intent→code, completions API | **19/20 (95%)** | código simples correto; logprobs ≈ 0 (certeza total) — filtro inútil |
| 30B, 20 intent→code, chat API mt200 | 17/20 (85%) | falhas = truncamento (docstrings longas); código mais idiomático |
| 30B, mt350 sem docstring | 17/20 (85%) | falhas: prosa vazada + corte mid-string + docstring malformada — **todas filtráveis por ast.parse**; ~10s/unit; Leibniz correta (*4) |

Conclusão: **logits não transferem entre arquiteturas diferentes** — o que
transfere é texto. O valor real do Qwen local é **diversidade em escala**
(ele cobre "código real variado" que meus 26 padrões synth não alcançam),
sob curadoria: eu desenho formato + filtro (ast.parse + heurísticas +
revisão), ele produz matéria-prima.

**Baseline decisivo (mesmas 12 intenções, n=4/intenção, prompt cru
`# intent\ndef`)**: modelos PRONTOS, zero treino nosso —
`Qwen3-0.6B` **83% aderência** (40/48) e `Qwen3-Coder-30B` 81% (39/48),
vs 61% do nosso `currv3` treinado. O 0.6B acertou até os domínios que o
teacher não cobriu (arquivo 4/4, binária 4/4). **Pretraining em trilhões
de tokens bate curriculum em 55M do zero** — o valor da nossa linha é o
pipeline e a controlabilidade, não a capacidade bruta.

## 7. O caminho esperado daqui

1. **curr v3 eval de intenção** — CONCLUÍDO: o canal existe.
   93/96 parse, **59/96 (61%) relevante**. Aderência segue cobertura do
   teacher: pi/frequência/soma/dicts/palindrome/async = 7-8/8;
   arquivo/busca-binária/média/ordena = 0-1/8. Canal real, cobertura fina.
2. **Corpus Qwen gerado** — `qwen_units.txt`: 139/148 units válidos (94%),
   16min, 30B-A3B. Domínios que faltaram no eval agora cobertos (IO, sort,
   estruturas, decorators, generators, regex, datetime, patterns).
   CAVEAT: docstrings e comentários "example usage" vazaram — strip antes
   de montar o curriculum v4.
3. Depois: **memória externa indexada** — o "GGUF de conhecimento" real:
   biblioteca grande e auditável que o modelo consulta, não memoriza. Os
   units `# ref: ...` + `# intenção: ...` já são o formato de consulta.
4. Remediações de medição (blind review RUN-047): decode non-ASCII
   (corrigido parcialmente), mascarar pads no CE, cache-key com hash.

## 8. Reprodução do zero

```bash
# 1. tokenizer + modelo: scripts/_bpe_probe.py (BPELM, ctx via env CTX)
# 2. corpus func_units: artifacts/ncm_code/func_units256.txt
#    treino base: podman run lcm-infer _bpe_probe.py TRAIN=1 UNITS=... CTX=256
# 3. teacher synth: scripts/_teacher_gen.py SEED=11 N_OUT=4000 → teacher_synth_v2
#    v3 intent:    scripts/_teacher_gen_v3.py → teacher_all_v3.txt
# 4. curriculum:  CKPT_INIT=gen_bpe_func256.leg2.pt UNITS=teacher_all_v3
#                 LR=1e-4 → ~320 steps
# 5. evals:       _eval_deep256.py (CTX=256!) · _eval_intent.py
# 6. REPL:        ./repl.sh gen_bpe_currv3.pt
# 7. Qwen probe:  llama-server -m <gguf> --port 8085 → /v1/chat/completions
```

Regras duras aprendidas: snapshot do ckpt antes de qualquer run que o toque;
eval sempre com `CTX=256` (senão shape mismatch); métricas de parse são
conservadoras (decode non-ASCII); NÃO usar `pkill -f` com padrão que case o
próprio shell.

## 9. Referências — o que usamos e para quê

Formato: referência — *papel que desempenhou no trabalho*.

### 9.1 Arquitetura e geração

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 (arXiv:1706.03762)
  — arquitetura do nosso 55M e de todos os professores.
- Radford et al., *Improving Language Understanding by Generative
  Pre-Training*, 2018 — modelo autoregressivo next-token; define o que
  "gerar" significa no REPL.
- Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023
  (arXiv:2211.19120) — KV cache e custo de decode; explica por que nosso
  `_bpe_probe` (sem KV cache) gera a ~1 tok/s e o llama.cpp a ~22 tok/s.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR
  2020 (arXiv:1904.09751) — nucleus sampling (top_p); nosso `topp09`.
- Ackley, Hinton, Sejnowski, *A Learning Algorithm for Boltzmann
  Machines*, Cognitive Science 1985 — origem da temperatura na amostragem.

### 9.2 Tokenização

- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with
  Subword Units*, ACL 2016 (arXiv:1508.07909) — BPE; nosso tokenizer
  byte-level de 8.192 merges (script próprio, `bpe_code_tokenizer.json`).
- Radford et al., GPT-2 report, 2019 — byte-level BPE (UTF-8→símbolos);
  fonte do nosso bug de decode non-ASCII e do mapeamento cp→byte.

### 9.3 Distillation (o eixo central)

- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*,
  2015 (arXiv:1503.02531) — KD clássica com logits suaves ("dark
  knowledge"); o que tentamos com o Qwen.
- Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016
  (arXiv:1606.07947) — KD por texto gerado; **é o que nosso teacher-data
  faz de fato** (logits → texto → CE hard).
- Boisvert et al., *Universal Logit Distillation Loss for LLMs*, 2024
  (arXiv:2402.12030) — logit-KD entre vocabulários diferentes via
  pooling sobre o vocabulário compartilhável; solução publicada pro
  problema que medimos inviável na forma crua.
- Zhang et al., *Dual-Space Knowledge Distillation for Large Language
  Models*, 2024 (arXiv:2404.13715) — DSKD: alinha logits entre
  tokenizadores por re-tokenização das saídas; o caminho "correto" se
  quisermos logits do Qwen no futuro.
- Gunasekar et al., *Textbooks Are All You Need* (phi-1), 2023
  (arXiv:2306.11644) — dados sintéticos de alta qualidade treinando
  modelo pequeno; **a inspiração direta da nossa estratégia teacher-synth**
  e a evidência externa de que código "de livro" bate corpus cru em
  modelos pequenos.
- Li et al., *Textbooks Are All You Need II: phi-1.5*, 2023
  (arXiv:2309.05463) — escala da mesma receita + o caveat documentado:
  synth demais restringe diversidade (nosso narrowing observado).
- Taori et al., *Alpaca*, Stanford 2023; Wang et al., *Self-Instruct*,
  ACL 2023 (arXiv:2212.10560) — professor gera pares
  instrução→resposta; análogo ao nosso canal `# intenção` → código (v3).

### 9.4 Currículo, esquecimento, replay

- Bengio et al., *Curriculum Learning*, ICML 2009 — teacher como fase
  final sobre base convergida (nossa receita B vencedora).
- McCloskey & Cohen, *Catastrophic Interference in Connectionist
  Networks*, Psychology of Learning and Motivation 1989 — o forgetting
  que medimos (+0.44 CE no FT puro).
- Goodfellow et al., *An Empirical Investigation of Catastrophic
  Forgetting*, 2013 (arXiv:1312.6211) — dropout/forgetting em redes
  modernas; base da leitura "shift de distribuição".
- Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural
  Networks*, PNAS 2017 (arXiv:1612.00796) — EWC; alternativa que NÃO
  usamos (preferimos curriculum — mais simples, medido melhor).
- Ratcliff, *Connectionist Models of Recognition Memory*, Psych Review
  1990 — rehearsal/replay; nosso probe mix-15% (e sua falha como dose).

### 9.5 Modelos e compressão

- Shazeer et al., *Outrageously Large Neural Networks: Sparsely-Gated
  MoE*, ICLR 2017 (arXiv:1701.06538) — MoE; arquitetura do Qwen3-30B-A3B
  (3B ativos/token → viável na CPU a ~22 tok/s).
- Yang et al., *Qwen3 Technical Report*, 2025 (arXiv:2505.09388) —
  família dos professores locais usados nos probes.
- *Qwen2.5-Coder / Qwen3-Coder Technical Reports*, 2024–25 — linhagem
  do 30B-A3B-Instruct (teacher de conteúdo nos probes §6).
- Lasby et al., *REAP the Experts: Why Pruning Prevails for One-Shot
  MoE Compression*, 2025 (arXiv:2510.13999) — o REAP-25B em disco;
  poda de experts vs quantização (contexto de escolha de teacher local).
- Leviathan et al., *Fast Inference from Transformers via Speculative
  Decoding*, ICML 2023 (arXiv:2211.17192) + EAGLE (arXiv:2401.15077) —
  os draft models em `~/models` (eagle3, dflash) e a rota de velocidade
  explorada em sessões anteriores do projeto.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR
  2022 (arXiv:2106.09685) — adapters como alternativa de fine-tune;
  considerada e não adotada (infra ausente no probe; curriculum resolveu).

### 9.6 Memória externa (a rota futura)

- Lewis et al., *Retrieval-Augmented Generation*, NeurIPS 2020
  (arXiv:2005.11401) — RAG; ancestral do "GGUF de conhecimento" real.
- Khandelwal et al., *Generalization through Memorization: kNN-LM*,
  ICLR 2020 (arXiv:1911.00172) — memória não-paramétrica no decode.
- Borgeaud et al., *Improving Language Models by Retrieving from
  Trillions of Tokens* (RETRO), ICML 2022 (arXiv:2112.04426) — retrieval
  dentro da arquitetura; referência pra fase `# ref:` futura.
- Ha, Dai, Le, *HyperNetworks*, ICLR 2017 (arXiv:1609.09106) — redes que
  emitem pesos; o beco técnico da ideia "gerar o GGUF direto" (§4).

### 9.7 Comportamento e opacidade

- Wei et al., *Chain-of-Thought Prompting*, NeurIPS 2022
  (arXiv:2201.11903) — a nuance honesta: hábitos de previsão produzem
  comportamento funcionalmente parecido com raciocínio.
- Elhage et al., *Toy Models of Superposition*, Anthropic 2022 — por que
  "onde está cada habilidade nos pesos" é pergunta em aberto.
- Carlini et al., *Quantifying Memorization Across Neural Language
  Models*, ICLR 2023 (arXiv:2202.07646) — memorização vs capacidade;
  enquadra o platô do leg-2 (train 0.46 / heldout ~0.93).

### 9.8 Software e infra efetivamente usados

- `llama.cpp` (ggml-org), build local `build/bin/llama-server`, API
  OpenAI-compatível `/v1/completions` e `/v1/chat/completions`,
  `logprobs`/`top_logprobs` — probes Qwen.
- GGUF: Qwen3-1.7B-Q8_0, Qwen3-0.6B-Q8_0, Qwen3-Coder-30B-A3B-Instruct
  UD-IQ2_M (topk4/topk6, Unsloth), Qwen3-Coder-REAP-25B-A3B-IQ2M,
  eagle3-145m-q8, coder-next-dflash-q8 — todos em `~/models/`.
- PyTorch + HF `tokenizers` (CPU), rootless Podman `lcm-infer:latest`,
  `--network none`, mounts `/repo` + `/work/artifacts`.
- Scripts próprios: `_bpe_probe.py`, `_teacher_gen*.py`, `_eval_*.py`,
  `_repl.py` — todos em `scripts/`.

## 10. Limites honestos atuais

- Sintaxe: saturada (32/32). Semântica: fraca — código plausível, não
  necessariamente correto/relevante (o que o REPL do usuário mostrou).
- `class ` e `def test_` argmax: buracos abertos documentados.
- Intent→code: **canal confirmado, 61% aderência** — limitado por
  cobertura de domínios no teacher, não por formato.
- Capacidade: 55M carrega hábitos, não fatos — memória externa é o caminho
  para "conhecimento em massa"; os pesos nunca serão a biblioteca.

## 11. O que sobrevive da linha BPE para a linha NCM (mapa de reuso)

A troca para BPE não foi desperdício — quase tudo transfere:

```
de BPE/curriculum          → para NCM-code
func_units256 corpus       → mesmo corpus alimenta o VQ-VAE/code rows
teacher v3 + qwen_units    → dados de curriculum do LevelGenerator
_eval_intent + parse/real  → métricas finais idênticas (comparável!)
ncm_panel.py (RUN-048)     → OS KNOBS DO RENDERIZADOR já existem
vq_code_tune.pt            → VQ afinado em código (teto ~25% medido —
                             esse número É o que precisa melhorar)
ncm_codetrain.py           → pipeline pronto, aceita EPOCHS/N_ROWS/MAX_T
método curriculum+snapshot → mesma disciplina de experimento
```

**Probes executados (sessão atual, ~1h cada):**

- **Probe 1 — `ncm_codetrain.py` bound** (EPOCHS=2, MAX_T=3600, ~220 steps
  a 16s/step): LevelGenerator oficial + VQ estrangeiro afinado. Loss
  desceu 6.36→5.71, mas parse 4/32→2/32 (ruído; ambos sopa de tokens).
  Confirma: **o gargalo é o canal VQ, não o gerador**.
- **Probe 2 — `train_vqvae.py` nativo de código** (config laptop5, dim 192,
  resume step 500→2100+, corpus tarefa→código): **held_ce 0.66 em step
  1200 e caindo** — o VQ-VAE nativo reconstrói código quase tão bem
  quanto o BPE prevê (0.92). `per_lvl_util[0]≈0.01` — l0 quase morto,
  casa com o achado anterior. **Veredito: o canal abre quando o encoder
  é nativo; o teto de 25% era encoder estrangeiro, não arquitetura.**

Próximo passo que isso destrava: LevelGenerator sobre os códigos DESTE
VQ (não o estrangeiro) — aí o `ncm_panel.py` vira renderizador de código
real, não de TinyStories.

### 9.9 Geração não-autoregressiva em escala (lacuna de prior art, 2026-02)

Análise externa (~/.gemini/antigravity-cli/.../analise-ncm-swe.md) apontou
omissão real: o log não tinha diffusion LMs em escala, que são o prior art
mais próximo da visão "renderizador de texto".

- Nie et al., *Large Language Diffusion with mAsking (LLaDA)*, 2025
  (arXiv:2502.09992) — LM de difusão mascarada 8B, 2.3T tokens; iguala/
  supera LLaMA3-8B em vários benchmarks; geração por refinamento
  iterativo com passes ajustáveis; edição não-monotônica; contorna a
  "reversal curse" de AR. Weights abertos (GSAI-ML/LLaDA-8B-Base).
- iLLaDA (2026, citado na análise externa; verificar fonte) — escala de
  ~12T tokens, competitivo com Qwen2.5-7B.
- *Diffusion of Thought* / MDLM / SEDD — difusão discreta de texto como
  linha acadêmica (a análise as cita como contexto).
- **Implicação para a tese**: o componente "geração por refinamento em
  passes" já existe em escala competitiva e roda em 8B Q4 (cabível na
  máquina). O diferencial original do projeto não é a matemática, é o
  **paradigma de controle**: painel com preview por nível, N amostras,
  samplers como "filtros", CFG, early-exit — `ncm_panel.py` implementa
  isso sobre NCM; portá-lo sobre LLaDA é alternativa com base pré-treinada.

### 9.10 Reframing da contribuição (análise externa, validado parcialmente)

O que o relatório externo acertou: (a) resultados negativos medidos do
log são o ativo publicável (parede DDR4 ~52GB/s, KD cross-vocab inviável,
posição-da-dose > tamanho-da-dose); (b) o nó "control paradigm" é o
diferencial, não a arquitetura; (c) zero artefato publicado/remoto é o
gargalo institucional; (d) cerimônia de governança desproporcional para
pesquisa solo.

Onde estava desatualizado ao ser lido: (a) baseline "decisivo" Qwen pequeno
já medido — 0.6B=83%, 30B=81% vs 55M=61%; (b) "teto VQ 25%" já falsificado
pelo VQ nativo (held_ce 0.42); (c) o "Caminho 3" recomendado é exatamente o
probe 3 em execução (LevelGenerator sobre códigos nativos).

Gap ainda aberto (concordância): eval por EXECUÇÃO do código gerado
(rodar testes), não apenas ast.parse — métrica correta para relevância.

### 9.9a Ecossistema dLLM verificado (pesquisa agy + checagem independente)

- **diffuse-cpp** (github.com/iafiscal1212/diffuse-cpp, Apache-2.0, base
  GGML): motor C++ de inferência para dLLMs mascaradas. GGUF F16/Q8_0/
  Q4_K_M para LLaDA-8B-Instruct e Dream-v0-Instruct-7B (org `diffuse-cpp`
  no HF). Entropy-exit (passes adaptativos 3–16) + inter-step KV cache.
  Sem tokenizer embutido (wrapper Python de ~15 linhas). Thread scaling
  medido pelo autor: **7.4× de 1→12 cores vs 2.4× do AR** — inferência de
  difusão é compute-bound (matriz×matriz), contornando a parede DDR4
  ~52GB/s medida aqui (§hardware). 14–28 tok/s em prompts fáceis num
  EPYC 12-core; prompts de código ~5 tok/s (16 passes).
- **Dream-7B** (arXiv:2508.15487, HKU+Noah's Ark, Apache-2.0): inicializado
  do Qwen2.5-7B, 580B tokens, CART (noise rescheduling adaptativo);
  melhor dLLM aberto em código/raciocínio. Q4_K_M ≈ 4.5GB → cabível.
- **LLaDA-8B** (arXiv:2502.09992, Apache-2.0): edição não-monotônica,
  infilling; Q4_K_M ≈ 5.2GB.
- DiffusionGemma (25B MoE): sem runtime CPU utilizável [unverified].
  Mercury/Gemini Diffusion: fechados.
- **Protocolo eval não-AR** (LLaDA §4, Dream §3): eixo X = NFE (passes de
  denoise; análogo a "passes de render"); eixo Y = pass@1 HumanEval/MBPP.
  Curva típica: S∈{16,32} colapso → {64,128} transição → {256,512}
  convergência. Análogo direto samples×ruído em ray tracing.
- Implicação estratégica: o "renderizador de texto" existe pronto em 7B
  Q4. O diferencial do projeto vira (a) painel de controle sobre esse
  motor e (b) medições qualidade×passes×tempo em CPU de consumo —
  contribuição ausente na literatura pública.

### Probe 3 — LevelGenerator sobre canal VQ nativo (veredito)

Setup: 62M params, codes_probe (11514×249, VQ nativo CE 0.42), 800 steps
batch 16, CPU, 4134s (~5.2s/step). loss 7.91→~5.35; l0 8.27→4.5;
níveis finos l2-l4 estagnados ~6.2-6.5 (perto de uniforme sobre 2048
códigos ≈ 7.6 bits).

Geração (final.pt, temp 0.8 topk40 e temp 0.3 topk10, l0 pela cabeça do
modelo): **0/32 ast.parse nas duas** — sopa de tokens, mesma falha do
canal estrangeiro MAS com diferença de mecanismo: **0.11s/amostra**
(6 passes de nível vs ~249 passos AR). O render paralelo funciona
mecanicamente; a coerência conjunta não emergiu em 800 steps/11.5k segs.

Leitura honesta: (a) canal nativo abriu (recon 0.42) — gargalo do encoder
resolvido; (b) o bottleneck passou a ser o LevelGenerator: níveis finos
carregam entropia local que 62M/1h não capturou (AR equivalente convergiu
a loss ~1.5-2.0 em 2000+ steps); (c) 800 steps é janela curta — veredito
é "insuficiente neste bound", não "arquitetura impossível".

Correções pendentes (receita agy, literatura SoundStream/EnCodec/
ViT-VQGAN): dead-code replacement + L2-norm de embeddings/codebook no
VQ (l0 util 0.01), depois retreino. Alternativa produto: painel sobre
Dream-7B/diffuse-cpp (compute-bound, foge da parede DDR4).

### Bound 1 — Overfit sanity check (veredito que fecha a linha NCM-from-scratch)

Setup: mesmo LevelGenerator 62M, mas 160 segmentos de código, 400 steps,
batch 32, 3905s. loss treino 7.84→0.83 (memorização profunda).

Geração: 0/32 ast.parse (temp 0.5). Amostras contêm fragmentos reais
("else :", "= mid - 1", "return - 1", "assert busca") em posições
incoerentes — o modelo reproduz estatísticas locais dos códigos, não
trajetórias latentes válidas.

Conclusão científica: (a) o render paralelo funciona mecanicamente
(0.11s/amostra, 6 passes); (b) a falha NÃO é de escala de dados/compute —
nem sob memorização total a geração produz código; (c) o defeito é
estrutural: níveis paralelos geram z fora do manifold do VQ (decoder só
vê códigos reais no treino) OU generate() diverge do condicionamento de
treino. Fechar linha NCM-from-scratch como resultado negativo
documentado; produto segue via diffuse-cpp/Dream-7B (§9.9a) que gerou
código AST-OK no primeiro disparo.

### Bound 2 — diffuse-cpp + Dream-7B no 4750U (primeira curva passes×qualidade)

Build em container llbuild (falha só em dump-logits, não crítico).
dream-7b-q4km.gguf 5.0GB. Wrapper scripts/dream_render.py (tokeniza
Dream/Qwen2.5 → cli --tokens → decode + trim em endoftext/im_end).

Primeiro disparo (is_prime, 4 threads sob contenção): AST-OK em 28
passes entropy_exit — mecanismo do renderizador validado.

Curva s∈{8,16,32} sobre prompt binary_search (-n160 -t8 temp0 argmax,
CPU livre, n=1/config → ruído):
- s=8: 8 passes reais, 73 tok, 47.5s, 1.54 tok/s — AST-OK mas bug
  semântico (left=left=mid+1, sem retorno de igualdade).
- s=16: 11 passes, 53.9s, 1.36 tok/s — AST-FAIL (tuple-assign bug).
- s=32: 11 passes, 53.4s, 1.37 tok/s — AST-FAIL (idem + token "&gt;").
entropy_exit termina antes do teto (8/11/11 passes reais).

Veredito honesto: pipeline funciona e é controlável, mas em código
Q4_K_M ~1.4-1.5 tok/s é 5-6× mais lento que AR denso de porte similar
nesta CPU (~8 tok/s para 7-8B Q4); qualidade argmax abaixo do AR
equivalente. Caminhos de melhora: Q8_0, temp>0+n>1, prompt tuning,
LLaDA-8B. O valor hoje é o paradigma de controle + infilling, não
velocidade bruta — e é compute-bound (escala com cores futuros).
