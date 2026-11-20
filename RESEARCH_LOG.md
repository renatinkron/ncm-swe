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

**O teste que falta** (a tese disruptiva de fato): `ncm_codetrain.py` com
bound curto (EPOCHS=2, MAX_T=3600) → LevelGenerator aprende código no
espaço VQ? Se o canal VQ é o teto (~25%), a pergunta certa é VQ-VAE
treinado DO zero em código (`train_vqvae.py`) em vez de reaproveitar o
de texto. Duas variantes, probes de ~1h cada antes de qualquer run longo.
