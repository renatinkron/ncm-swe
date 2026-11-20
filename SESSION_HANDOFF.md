
## Sessão: build master + hipótese OpenMP barrier (2025)

### Contexto
Profile anterior: 60.4% CPU em libgomp (barrier/spin) durante decode do 80B.
Hipótese: GGML_OPENMP=ON faz ggml_barrier -> libgomp; issue ggml-org#26200
documenta +40% em mingw ao compilar com GGML_OPENMP=OFF.

### Builds
- Fonte: clone oficial ggml-org/llama.cpp, commit 5676dfb (master)
- Container build: imagem `llbuild` (gcc14+cmake), `podman run -v src:/src`
- Variantes: `build/` (GGML_OPENMP=ON, default) e `build-nomp/` (OFF)
- Binarios copiados para `/home/kron/.local/opt/llama-master-nomp/`

### A/B medido (mesmo arquivo, mesma maquina, r=3)

**30B IQ2_M-topk6 (residente 10.3GB):**
| build | tg32 t/s |
|---|---|
| b10107 (OpenMP) | 20.25 |
| master (OpenMP) | 20.06 |
| master nomp | 20.07 |
-> Zero diferenca no 30B: libgomp barrier ja faz spin no Linux.

**80B IQ1_M-topk5 (arquivo esquentado em serie):**
| build | tg32 t/s |
|---|---|
| b10107 (OpenMP) | 10.16 +- 0.34 |
| master nomp | 10.41 / 10.85 +- 0.2 |
-> Ganho ~+3-7%, variancia menor. Nao e a bala de prata.

### Conclusao corrigida
Os 60% em libgomp NAO sao custo de mecanismo de barreira - sao threads
esperando a cadeia sequencial de ~4500 nos/token (straggler + latencia serial
entre ops dependentes). OpenMP=OFF ajuda marginalmente (spin vs futex).
O gargalo real = numero de nos no grafo qwen3next (GDN+MTP+MoE hibrido
duplica nos vs MoE normal). Alvo seguinte: fusao de ops / reducao de nos.

### Rejeitado nesta sessao
- OMP_WAIT_POLICY=ACTIVE: medido 3.13 t/s mas com build em background
  disputando CPU - numero sujo, nao conclusivo.
- troca de mecanismo de barrier como alavanca principal.

### Estado operacional
- Servidor :8089 = 80B topk5 + DFlash n_max=3, build llama-master-nomp
- pi configurado (~/.pi/agent/SYSTEM.md + settings.json, provider local)
- baloo_file_extractor suspenso (estava indexando GGUFs, comendo cache)

### Proximo passo promissor
Fusao de ops no grafo qwen3next: o modelo tem ~2x nos/token vs MoE normal.
Candidatos: norm+gate, views em build_qkvz, adds residuais sequenciais.
Sem flag de profiling por-no nesta versao - instrumentar ou gprofng.

## Sessao: auditoria completa de engines e modelos (2025)

### Per-node profiling (patch GGML_NODE_PROF em ggml-cpu.c)
Breakdown de parede por token no 80B topk5 (~113ms/token):
- MUL_MAT 57.5ms (433 nos) | MUL_MAT_ID 35.3ms (144) | GET_ROWS 7.8ms
- GDN 4.8ms | CPY 3.4ms | resto ~4.7ms
- MUL_MAT ~52GB/s efetivo = TETO DDR4-3200 (~51GB/s). Bandwidth-bound real.
- VIEW/RESHAPE/PERMUTE/TRANSPOSE (1764 nos) sao metadados, pulam barreira.

### Engines testadas (mesmo arquivo, quente)
| engine | 80B topk5 tg | 30B topk6 tg |
| b10107 | 10.16 | 20.25 |
| master-nomp | 10.4-10.9 | 20.06 |
| ik_llama | 10.69 | 18.17 |
-> Paridade. Os 3x do ik_llama no Ryzen-3995WX = maquina de ~200GB/s, nao se aplica aqui.

### Spec sweep
- DFlash n_max=3: campeao (~10.4 efetivo, 73-82% acc)
- n_max=4/6/15: pior. Combo dflash+ngram-mod: piorou (28% acc)
- draft-mtp: MORTO - Coder-Next nao tem tensores mtp.* (Instruct tem, shard 41)
- EAGLE3 145M: 64% acc, contexto draft 4k

### Modelos alternativos testados
- REAP-25B (podado): 15.5 t/s - mais lento que 30B cheio
- 35B-REAP qwen35moe (nova gen): 10.98 t/s - empata com 80B
- Qwen3.8-Flash-Next: 55-68GB min quant - inviavel (2x RAM)

### Pendente (poucos %, requer root)
- governor schedutil->performance, THP madvise->always

### Conclusao
Hardware no envelope: ~50GB/s DDR4 = 10.5 t/s no 80B, 20 t/s no 30B.
Teto fisico atingido; ganhos adicionais so via hardware maior ou modelo menor.

## Sessao: DFlash draft para o 30B (2026)

### Descoberta chave
Existe draft DFlash OFICIAL treinado para o Qwen3-Coder-30B-A3B:
- z-lab/Qwen3-Coder-30B-A3B-DFlash (autores do paper DFlash)
- 0.5B params, 8 camadas, d=2048, block_size=16, mask_token 151669
- Treinado em 289K amostras (Nemotron code + evol-codealpaca + Cline traces)
- GGUFs prontos: AtomicChat/Qwen3-Coder-30B-A3B-DFlash-GGUF (Q8_0, 509MB)
- Baixado: /home/kron/models/qwen30b-dflash-q8.gguf
- Conversao GGUF para draft standalone suportada: convert_hf_to_gguf.py --target-model-dir

### Medido na maquina (DDR4 dual-channel, bandwidth-bound)
| config | tg t/s | acceptance |
| 30B sozinho (ngram-mod) | 19.16 | - |
| 30B + DFlash n_max=3 | 17.5-19.0 | 77-90% |
| 30B + DFlash n_max=4 | 14.6 | 62% |

Veredito: EMPATE. O draft e de alta qualidade (77-90% acceptance) mas
na maquina bandwidth-bound o custo se anula: o verify pass do MoE le
~n_max+1x os bytes de experts (cada token proposto puxa seus experts)
e o draft custa ~0.5GB/bloco. No 80B funcionou (+18%) porque o alvo
era 2.3x mais lento por pass - overhead do draft diluia.

### Implicacao
- Nesta maquina: 30B sem draft continua otimo (~19-20 t/s)
- Em hardware bandwidth-rico: o MESMO draft daria +50-80% - o limitante
  e a maquina, nao o draft
- Draft fica no disco como opcao documentada

### Ecossistema DFlash mapeado
- Treino: SpecForge (z-lab) e NeMo AutoModel tem receitas DFlash
- Arquitetura: draft de bloco-difusao condicionado a hidden states do
  alvo em target_layer_ids, compartilha embed/lm_head do alvo congelado
- Runtime llama.cpp: --spec-type draft-dflash (master-nomp OK)
- Variante TEV/ExitTrain existe (hsj576) mas e GPU-only

### Estado operacional
- :8089 = 30B topk6, ngram-mod, master-nomp (config pre-dflash restaurada)
- /home/kron/models/qwen30b-dflash-q8.gguf disponivel para re-teste

### Contexto NCM (discussao)
- l0-sampling do recipe oficial JA implementado e testado no modelo limpo:
  dup_rate 0.22->0.11 mas textos seguem com gagueira - nao e a chave
- Gap restante = orcamento de treino do condicionamento intra-nivel
  (oficial usou ~2000x compute)
- Caminho "converter treino em decode paralelo" que CABE na maquina:
  draft heads sobre trunk congelado (regime forward-only). Instanciado
  na pratica pelo DFlash - cuja economia nao fecha nesta banda.

## Sessao: "olhando o lado errado" - auditoria hardware + KV (2026)

### Hardware medido pela primeira vez
- CPU: AMD Ryzen 7 PRO 4750U (Zen2, 8c/16t, AVX2, sem AVX-512)
- Banda real medida: read 36.7 GB/s, triad 19.3 GB/s (8 threads)
  = dual-channel DDR4-3200 saudavel (~72% do teorico 51.2) - SEM defeito
- THP=always ja ativo, governor=schedutil (requer root p/ mexer)
- Build master-nomp TEM AVX2+FMA (20.9k instrucoes vfmadd/ymm)

### Modelo de duas paredes (NOVO DIAGNOSTICO)
- Parede 1 (batch=1): BANDA ~37GB/s -> cap ~20 t/s no 30B IQ2M
- Parede 2 (batch>1): COMPUTE ~26 GFLOP/pass no Zen2 -> spec decode
  troca uma parede pela outra, por isso DFlash empata

### Quants testados (A/B real, arquivo quente)
| quant | bpw | tg t/s | regime |
| IQ1_S | 1.56 | 3.88 | compute-bound: dequant pesado mata |
| IQ2_M-topk6 | 2.7 | 20.35 | SWEET SPOT |
| Q3_K_XL | ~3.4 | 16.62 | bandwidth-bound: bytes demais |
-> IQ2_M ja e o ponto otimo da maquina. Abaixo, kernel de dequant
   nao acompanha; acima, bytes demais. IQ1 nao serve pra CPU fraca.
- Arquivo IQ1_S em /home/kron/models/qwen30b-iq1s.gguf (morto, apagar?)

### ACHADO: contexto degrada MUITO mais que o modelado
Curva limpa no 30B (decode t/s):
- ctx ~30: 19.4 | ctx 3.3k q8KV: 6.49 | q4KV: 7.89 | f16KV: 4.71
- f16 KV aloca 25.8GB em -c 131072 -> thrash total (modelo evictado)
- Custo real do KV/token ~24x o teorico (GQA expansion + compute attn)
- FA on = auto (ruido). O( ctx ) attention na CPU e o vilao: em ~37k
  ctx o KV custa mais que o modelo inteiro

### Config campea final (:8089)
- 30B IQ2_M-topk6 + --cache-type-k/v q4_0 + -fa on + jinja
- q4 KV: +21% a ctx>3k e metade da RAM (6.4GB vs 12.9GB alocados)
- ngram-mod NAO registra nesse build - removido da config
- DFlash 30B baixado mas empata na maquina (ver sessao anterior)

### Recomendacao operacional
- Sessoes curtas ou restart do servidor entre tarefas (KV reseta)
- pi: manter contexto compactado; >30k ctx = degradacao severa
- -c 131072 nao e inocuo: KV alloc reserva RAM e leitura cresce O(ctx)

## Sessao: RUN-035 — probe l0 via features de LLM congelado (NEGATIVO)

Pergunta (hipotese de conversao LLM->NCM): features do Qwen2.5-0.5B
congelado (~models/qwen05-sft, backbone do RUN-005) predizem o codigo l0
(1 posicao, 4096 classes) do tokenizer oficial? Setup: prefixo de 16
tokens BPE de cada segmento -> feature -> probe.

### Pipeline implementada (nova, em scripts/)
- dump_l0cond_rows.py (container): reproduz segs_of do encode_official
  EXATAMENTE (SHARDS=25 -> 125k records -> 177175/15400 rows, batendo
  meta.json) + emite prefix_ids/prefix_text por code-row
  (artifacts/l0cond/rows_{train,held}.jsonl)
  - BUG cuidado: U+2028 dentro de prefix_text quebra jsonl por linha —
    ler com raw_decode (iter_rows em l0cond_features.py)
  - corpus_ts_vq sao symlinks que quebram no container; usar
    CORPUS=/repo/data/corpus_ts
- l0cond_features.py (host ~/.venvs/ncm): FrozenQwen reusado de
  train_level_heads.py; arm E = mean(embed_tokens) todos rows; arm H =
  mean-pool hidden (subsample; forward 0.5B ~ 1 TFLOP/batch64)
- l0cond_probe.py: {linear,mlp} feature->4096 + baselines
  (uniform/marginal-com-alpha/shuffle) -> artifacts/l0cond/probe.json
- bench_l0cond.py (container): geracao pareada l0=adapter vs l0=unigram
  nos mesmos prefixes -> artifacts/bench_l0cond.json
  - NOTA: sample_official.py roda main() no import — nao importar dele

### Resultados (artifacts/l0cond/probe.json)
Baselines: uniform 8.318 | marginal 7.800 (coverage held 99.9%)
| braco | held CE | leitura |
| E_linear | 7.867 | == E_shuffle 7.867 -> so aprendeu a marginal |
| E_mlp | 7.801 | = marginal exato |
| H_linear | 10.638 | ~= H_shuffle 10.667 (overfit: 8k rows/4096 cl) |
| H_mlp | 13.900 | overfit pior |
bench (n=16, pareado apos fix do blind review): adapter rank mediano
do l0 real = 1685/4096, top8=0; geracao adapter cos 0.550 vs unigram
0.557 — empate dentro do ruido (adapter == amostrador da marginal).
Blind review: fix RNG pareado por braco + max_length 24=extracao.

### VEREDITO — H1/H2 REJEITADAS
Features congeladas de prefixo NAO predizem l0 alem da marginal
(probe com label real == controle shuffle). Consequencia cientifica
importante: o l0 do VQ oficial NAO carrega "conceito" recuperavel por
features semanticas — confirma e afia o diagnostico RUN-015/016
("residual VQ fatora por erro de reconstrucao, nao previsibilidade").
O l0=0.94 do RUN-005 era real mas num codebook SEMANTICO (piramide
sobre embeddings) — objeto diferente. Corolario: quer nivel conceitual
predizivel? CONSTRUA o codebook pra isso (ancoragem semantica),
nao espere que um VQ de reconstrucao o forneca.
Unigram do recipe oficial segue sendo o tratamento correto p/ l0.

---

## RUN-036 — Destilação conversacional com meta-ensino (0.5B aluno ← 30B professor)

**Data**: 2026-11-13 | **Status**: completo — resultado rico (positivo+negativo documentados)
**Scripts**: `scripts/convdistill.py` (driver), `scripts/kpack.py` (catálogo+transplante),
`scripts/chat_student.py` (REPL do aluno), `scripts/watch_convdistill.py` (viewer)
**Artifacts**: `artifacts/convdistill/` (metrics/transcript/livechat.jsonl,
eval_frozen/knowledge_pack/kpack_eval.json, kpack_index.npy, student_final.pt ~2GB)

### Setup medido
- Professor: llama-server :8089 30B IQ2_M-topk6 (q4 KV, fa on) — OpenAI API
- Aluno: qwen05-sft (Qwen2-0.5B) in-process, **full FT fp32**, 494M params, AdamW lr=1e-5, clip 1.0
- Loop/rodada: professor emite JSON {skill,probe,ideal,expected_tokens,strategy} →
  aluno pré-teste (rubrica) → SFT step → re-probe retenção → feedback ao professor
- 24 rodadas, 27.1min, ~50-85s/rodada (teacher ~14s + sft ~4s + gens)

### Resultados (metrics.jsonl)
| eval | ce_teacher | ppl_neutral | drift |
|---|---|---|---|
| baseline | 0.7659 | 29.41 | 0 |
| @6 | 0.7332 | 28.69 | 0.058 |
| @12 | 0.7338 | 28.36 | 0.107 |
| @18 | 0.7651 | 28.24 | 0.151 |
| @24 | 0.7766 | 28.46 | 0.203 |
- Retenção (re-answer de item falho pós-1-step): **0.5 → 0.667** (r14→r22)
- Professor adaptou visivelmente (strategy_note referencia retenções; escalou só após reter)

### Achados (o que importa)
1. **Meta-ensino funciona**: professor aprendeu a calibrar dificuldade ao aluno;
   retenção subiu mensuravelmente durante a conversa.
2. **Reward hacking observado** (achado central): o professor otimizou a métrica
   que via (retenção) → ensinou só itens fáceis → ce_teacher em eval GERAL
   *regrediu* (0.766→0.777). Narrow SFT melhora o nicho treinado à custa de
   generalidade. Lição: a métrica de feedback molda o currículo.
3. **Sem forgetting**: ppl_neutral 29.4→28.5 (estável/melhorou) em 24 steps.
4. **Conhecimento entra nos pesos**: aluno produziu `add_numbers` (nome inventado
   pelo professor, não estava no prompt) com args generalizados (x,y vs a,b).
5. **Catálogo kpack.v1**: pack de 24 cards + índice embed funciona mecanicamente
   (retrieval correto), MAS teste de transplante **saturou** (8/8 naked=armed=1.0 —
   itens fáceis demais, rubrica de assinatura trivial). Inconclusivo; débito:
   eval com itens onde naked falha.
6. **Gargalo real do 0.5B**: loop de repetição (emite im_end como texto, não
   para) + context-locked (correção conversacional ignorada; pergunta fora de
   domínio → regressa ao 1º tópico). Visto ao vivo em livechat.jsonl.
7. Teacher morreu no fim da sessão (PID kill manual a pedido do user p/ liberar RAM).

### Protocolo compacto (não medido)
- Teste proposto (professor inventa notação compacta pros cards, aluno decodifica)
  ficou pendente — servidor derrubado antes. Artefatos prontos p/ retomar.

---

## RUN-037 — Cartuchos de domínio plugáveis (prova do conceito do "download dirigido")

**Data**: 2026-11-13 | **Status**: completo — **POSITIVO**: prova end-to-end
**Script**: `scripts/domcart.py` | **Artifacts**: `artifacts/domcart/`

### Ideia testada (visão do user)
Um modelo generalista recebe "download" de uma área específica como módulo
plugável: especialista com cartucho, generalista intacto sem ele. Implementado
como **LoRA r=8 por domínio** (arquivo separado, base nunca sobrescrita).

### Setup
- Domínio: Python regex; rubrica por EXECUÇÃO real (pattern compilado e
  rodado em subprocess contra 33 asserções, timeout 8s)
- Professor 30B gerou 24 lições (skills: char classes→sub); aluno treinou
  2 epochs LoRA (4.4M params, ~0.9% do modelo), base congelada

### Resultados (metrics.jsonl — números reais)
| condição | pass_rate | ppl_neutral |
|---|---|---|
| base | 0.500 | 29.41 |
| **plugado** | **0.765** (+53%) | 28.26 |
| desplugado | 0.500 | 29.41 (idêntico) |
- Cartucho: `cartridge_regex.pt` = **17.7MB** — esse é o "download"
- Plugado emite regex de verdade (`[\w\.-]+@[\w\.-]+\.\w+`); base ecoava a tarefa
- Reversibilidade EXATA: unplugged reproduz baseline à risca
- Sem forgetting (ppl neutro não regride; até melhora leve plugado)

### Veredito
Sim, é possível — tier provado nesta máquina: mesma-arquitetura+LoRA.
Próximos degraus honestos: (a) multi-cartucho (n packs, swap por demanda),
(b) tiers — texto kpack (universal, RUN-036) vs LoRA (forte, por-arch),
(c) composição de cartuchos (stacking LoRAs — interferência é aberta).

---

## RUN-038 — NCM como estudante do knowledge pack (prova cross-arquitetura)

**Data**: 2026-11-13 | **Status**: completo — **POSITIVO no nível de distribuição**
**Script**: `scripts/ncm_packtrain.py` | **Artifacts**: `artifacts/ncm_packtrain/`

### O teste
O pack_regex (24 cards, gerado pelo 30B pro Qwen-0.5B em RUN-037) foi codificado
nos códigos VQ oficiais do NCM e usado pra FT leve (30 epochs, lr 1e-4) do
LevelGenerator gen_ncm_v2 — prova de que o pack porta conhecimento entre
arquiteturas radicalmente diferentes (transformer-token vs VQ-LCM).

### Resultados (metrics.jsonl)
- train_ce 8.19 → 3.21
- regex_score/segmento: **0.00 → 0.38** | compilável: **0/16 → 4/16**
- PRE: TinyStories puro; POST: vocabulário regex-domain ("pattern matches",
  "re.", "a string letter") — deslocamento claro, qualidade limitada pelo
  teto do NCM (word salad esperado).

### Veredito
O formato de pack (texto+rubrica) é realmente model-agnostic: o mesmo artefato
ensinou três estudantes diferentes — 0.5B full-FT, 0.5B+LoRA cartridge, e NCM.
Limite honesto: pack move distribuição, não fluência — o NCM precisa de um
gerador melhor pra colher o conteúdo como texto usável.

## RUN-038 (NCM pack transfer)
- `scripts/ncm_packtrain.py`: FT in-memory do LevelGenerator nos cards regex do pack via tokenizer oficial. Dist shift: hits 0->0.38, 4/16 compilaveis (vocab regex aparece, fluencia nao). Ckpt intacto.
- Diagnostico novo: VQ-VAE perde 26% tokens de codigo na reconstrucao (2.0 ch/tok, story=4.2/100%) -> proximo passo: BPE de codigo.

## RUN-039 — NCM aprendendo código (em curso, máquina dedicada)

Contexto: user pediu ensino REAL ao NCM (refinar até programar). Diagnóstico
prévio medido: VQ-VAE oficial perdia 26% dos tokens de código (rec 73.9%).

Pipeline montada:
- `scripts/code_corpus.py`: 168MB Python real (stdlib+site-packages+repo) ->
  data/corpus_code/ (166 shards)
- `scripts/code_bpe.py`: BPE-8192 de código -> artifacts/ncm_code/bpe_code_*;
  densidade 1.68 -> 2.75 chars/tok em código
- `scripts/code_recon.py`: gate do canal. recovery: 65.2% mismatch ->
  67.0% só-emb -> **96.4% emb+decoder tuned** (enc/VQ congelados).
  Artefato: artifacts/ncm_code/vq_code_tune.pt (wte+decoder state)
- `scripts/ncm_codetrain.py`: encode corpus -> codes_code.npy (20000,641)
  cacheado; FT cópia do LevelGenerator (ckpt original intocado); amostras
  vivas a cada 5 épocas em artifacts/ncm_code/live_samples.json; métricas
  parciais em train_metrics_partial.json; eval final ast.parse+compile.

Execução: podman lcm-infer, N_ROWS=20000 EPOCHS=20 THREADS=14, batch=16,
lr=1e-4, ~1250 steps/época (~30-40min/época -> ~10h total).
Log: /tmp/ncm_codetrain.log. Baseline: parse 0/32, compile 0/32.

Medição chave já registrada: benchmark de geração = 137.2M params totais,
576 códigos/s AR, decode ~grátis => ~110 text-tokens/s equivalente, ~550MB
fp32. A tese "modelo de desktop" é confirmada em throughput; falta qualidade.

Próximo: observar live_samples (estrutura de código vs word-salad), CE curve,
parse% final. Se fluência sair: corpus de trajetórias (problema->código->
teste->correção) é o degrau seguinte rumo a comportamento de agente.

### RUN-039 resultados (sessão longa, máquina dedicada)
- Pipeline código: corpus 168MB -> BPE-code (2.75 ch/tok) -> VQ tune (recovery 96.4%)
- bootstrap currículo próprio (127 micro-lições, scripts/teach_corpus.py):
  CE 6.10->0.34 c/ EOS fix (gen_lessons2.pt). Lições completas ensinam
  estrutura; parse_post=0/32 msm temp 0.3 — corrupção de superfície
  (~4% canal + níveis finos) quebra sintaxe Python. Conteúdo aprende,
  realização não sustenta gramática rígida nessa escala.
- kdistill (seleção densa, scripts/ncm_kdistill.py): 200 rows top-loss +
  dedup l0 -> heldout CE 7.93->6.75 em 150 steps. Braço random pendente.
- Bench throughput: 137M params, ~110 tok/s equiv, decoder ~grátis.
- kdistill VEREDITO: random 6.63 < condensed 6.75 (base 7.93). Seleção por
  loss REFUTADA nesta escala — dificuldade != transferibilidade. Resta:
  diversidade de gradiente (caro), decoder>99% (única alavanca medida).

### RUN-039 fechamento — decoder boost + diagnóstico de falha (2026-11-14)

INCIDENTE + regra nova: `dec_boost.py` rodou com `vq.train()` (dropout ligado)
e sobrescrevia `vq_code_tune.pt` cegamente a cada 500 steps — o checkpoint de
96.37% virou 90.24%. Fix: `vq.eval()`, save só-em-melhora, LR 1e-4, snapshots
timestampados. **Regra permanente adicionada em AGENTS.md**: snapshot antes de
qualquer run que escreva em checkpoint existente + save gated por métrica.

Decoder boost corrigido (2500 steps, 99min, emb+decoder, enc/VQ congelados):
recovery 90.24 -> **93.33%** (hist: 91.82/92.63/93.07/93.33/93.32 — platô).
Artifacts: `dec_boost_hist.json`, `vq_code_tune.pt` (best), `.bak.pt`,
`.best_20261114_0928.pt`.

DIAGNÓSTICO CHAVE (`scripts/_diag_channel.py`): lições REAIS -> encode -> VQ
-> decode: **ast.parse 25/127 (19.7%)**, tok recovery 98.33%. Ou seja: mesmo
com códigos perfeitos (zero erro do gerador), ~80% das lições morrem no parse
— Python não tolera ~1.7% de corrupção de token. FIRST_BAD típico: docstring
e return perfeitos, 1 token corrompido na assinatura (`total_three8,,`).

CONCLUSÃO DO RUN (negativo, bem localizado):
1. Canal reparado 65->93% (98% em lições) — mas o platô sugere **teto
   intrínseco do codebook VQ** (<99.9% que Python exige), não só decoder.
2. Gerador aprende conteúdo/estrutura (CE 0.34, if/elif/docstring/while
   emergem) mas seus 0/32 são a soma canal+gerador — secundário ao canal.
3. ast.parse final: 0/32 (melhor caso, após boost, temp 0.3/0.6: 0/16).
4. kdistill: blind-review F1 corrigiu o wording — o scoring era por
   CHUNK de 32 rows, não por row. **Chunk-mean selection refutada**;
   per-row selection segue NÃO testada (precisa reduction='none', ~32×
   mais forwards).

Alavancas restantes, por custo:
a) denoiser/repair head pós-decode (corrige `three8,,`->`three(`) — barato;
b) retreinar VQ com codebook maior — invalida codes+gerador (recomeço);
c) aceitar teto: ~110 tok/s de "quase-Python" — ativo real, gramática não.
Próxima sessão: escolher entre (a)/(b)/(c) com o user.

## RUN em curso — decoder em escala + gerador no corpus (2026-11-15)

INSTRUÇÃO PERMANENTE DO USUÁRIO (2026-11-15): **manter a sessão de
monitoramento sempre ativa, mesmo ao responder perguntas do usuário** —
o acompanhamento do treino NUNCA para por causa de conversa paralela.

Decoder boost em escala (dec_boost3.py, 49k rows, 6000 steps, vq.eval,
save só-em-melhora): recovery 93.33 -> **96.80%** (monotônico:
94.05/94.80/95.34/95.83/96.23/96.36/96.55/96.67/96.74/96.78/96.80).
Checkpoints: vq_code_tune.pt (best) + snapshots timestampados.

TESTE DECISIVO — overfit decoder em 64 rows held-out: **100% rec @200
steps** => códigos VQ são LOSSLESS (determinam tokens); platô anterior
era GENERALIZAÇÃO do decoder (dados), não perda intrínseca do codebook.
Refuta a hipótese "teto intrínseco" registrada na conclusão do RUN-039.

Canal-only pós-boost (lições reais -> encode -> VQ -> decode): ast.parse
25/127 -> **71/127 (55.9%)**, tok_rec 99.37%. Erro típico residual:
1 token corrompido por lição.

Gerador teacher-forced nas lições (64 rows): l0 7.8%, l1-l7 88-96%,
l8-l11 67-83% — fine acc 81.7%, 0 rows com 641/641 códigos certos =>
0/16 explicado: bloqueio restante é o GERADOR (níveis finos/superfície).

TREINO DO GERADOR EM CURSO (ncm_codetrain.py, podman lcm-infer,
--network none): INIT=gen_lessons2.pt, N_ROWS=8000, EPOCHS=15, BATCH=16,
LR=1e-4, THREADS=14->cap 12 (renice +15), GEN_OUT=gen_code2.pt.
Instrumentado: STEP_LOG=50, live_samples.json, train_metrics_partial.json,
save ATÔMICO por época (.tmp->rename). CE: ep0 5.165 -> ep1 4.504 ->
ep2 4.184 -> ep3 3.895 (em curso). ~15.5s/step, ~2.15h/época.

INCIDENTE 2 (menor): run anterior de 8k completou 2 épocas (CE 4.347,
live_samples epoch:1 @08:20) e foi morta pelo relaunch às 08:41 —
checkpoint sobrescrito, ~2 épocas perdidas. Lição: verificar artifacts
escritos ANTES de kill, não só o log.

PRÓXIMA AÇÃO: quando ep4-6 fecharem, re-medir acurácia níveis finos
(meta >=95% p/ parse) e rodar ast.parse nos samples. Se fino estagnar
<95%: repair head pós-decode (plano B) ou escala do gerador (plano C).

## MEDIÇÃO 2026-11-16 — acurácia por nível no corpus de código (decisiva)

Setup: `_gen_acc.py` parametrizado (GEN_CK, CODES, OFFSET). Teacher-forced,
exact-match por nível, 64 rows held-out de codes_code.npy (OFFSET=8000;
treino usa rows 0:8000). VQ congelado = vq_code_tune.pt.

Held-out code rows (baseline gen_lessons2.pt vs gen_code2.pt @~ep4-5):
- fine levels overall: **4.29% -> 10.16%** (2.4x; generaliza)
- l1: 10.7% -> **44.9%** (nível fino inicial aprende mais rápido)
- l2-l6: ~1-6% -> 2.7-6.2% | l7: 1.6% -> 10.5%
- l8-l10: ~3-5% -> 7.3-9.9% | l11: 7.2% -> **17.9%**
- l0: 0% (conceito grosso nunca acerta exato em held-out — entropia
  intrínseca do canal, não só fraqueza do gerador)
- rows com 641/641 códigos: 0%

Rows de TREINO (OFFSET=0, mesmo checkpoint): fine 18.3% (l1 50.8%,
l7 20.9%, l11 28.7%). Gap treino/heldout 18.3 vs 10.2 = moderado;
modelo majoritariamente SUBAJUSTADO (capacidade/entropia), não
superajustado.

Distribuição da deficiência: DIFUSA — todos os níveis finos fracos em
código real (diferente das lições, onde era l8-l11 localizado).
Implicação: loss ponderado por nível teria retorno pequeno; não há
cluster isolado de deficiência. Treino dirigido por "hard levels"
NÃO priorizado.

CE ~3.2-3.6 (codebook ~512, random ln512=6.24) mapeia pra ~10-18%
exact-match => cada ponto de CE compra menos acurácia; trajetória
para >=95% fino por esta via é longa. Estratégia favorecida pela
medição: decodificação (best-of-N + ast.parse + repair head) em vez
de exigir acerto exato 641/641.

Otimizações do run (medidas): torch.compile ~=0% (roofline: ~450-475
GFLOPS sustentados ~= 90-100% do pico fp32 Zen2); pinning em 8 núcleos
físicos via sched_setaffinity = mesmo ritmo com menos CPU; tcmalloc
~=nulo; renice +15. Podman --cpuset-cpus falha em rootless cgroup v2.
Vide agent reniced +10; sleep órfão antigo morto.

## RUN-040 — higiene de artefatos + livro de metodologia (2026-11-16)

INSTRUÇÃO DO USUÁRIO: guardar todo material e registros, mas NÃO todos os
artefatos — disco enchendo e precisamos de espaço para treinar. Em
paralelo: montar livro interativo sobre a metodologia de pesquisa do
projeto (lançar ao fim do "caminho probatório").

LIMPEZA: artifacts/ 49G -> 17G (**32G liberados**). Política permanente +
manifesto em `docs/ops/ARTIFACT_CLEANUP_RUN-040.md`: registros/métricas/
código/final.pt preservados; ckpt.pt e ckpt_step*.pt de linhas encerradas
e runs de profiling/sanity deletados. Protegidos confirmados íntegros:
ncm_code/ inteiro, gen_ncm_v2/final.pt (fonte de config dos scripts),
ncm_official/, convdistill/student_final.pt. Run ativo inalterado.

LIVRO: `course/` criado — README + 11 módulos (m00-m10), todos os labs
citando artefatos/medições reais do repo. Cobre: ciclo hipótese→evidência,
instrumentação, decomposição de pipeline, métricas honestas,
roofline/profiling, governança de recursos, negativos como ativos,
handoff, decodificação best-of-N+repair, capstone com próxima barreira
(capacidade). Validado: links internos OK, caminhos citados existem.

## MEDIÇÃO 2026-11-16b — CE acelera + top-k decompõe o ganho

Trajetória CE estendida (run otimizada retomou do ep3):
5.165→4.504→4.184→3.895→3.609→3.244→2.878→2.516→~2.2 (ep4/global-8
em curso). Descida ACELERANDO (~-0.36/época recente).

Re-medição em gen_code2.pt @CE2.5 (held-out 8000:8064):
- fine top1: 10.16% -> **11.79%** (CE -1.1 ponto rendeu só +1.6pp)
- **fine top5: 22.14%** | l1 top5 **69.7%** | l11 top5 36.7%
- l0 segue 0%; rows 641/641: 0

Samples ep3: vocabulário Python REAL emergiu (def/return/except/
isinstance/docstrings) vs salada anterior — superfície ainda corrompida.

Leitura: ganho de CE concentra-se em massa de "quase-acerto" (top-5
dobra o top-1) => confirma (a) que exact-match por argmax é beco longo,
(b) que SAMPLING colhe ~2x mais material que argmax sugere — dado-chave
p/ dimensionar best-of-N (p efetivo por posição ~22%, não ~12%).


## MEDIÇÃO SAMPLE+PARSE 2026-11-16 (gen_code2 @ ep10 CE 1.7108)

Free-running generation eval (scripts/_sample_parse.py, container lcm-infer, THREADS=4):
- temp0.9/topk32: parse 0/32, compile 0/32 (31s/32 samples)
- temp0.5/topk16: 0/32
- temp0.3/topk8:  0/32
- temp0.1/topk4:  parse 1/32 — DEGENERADO (1 stmt = bare name + docstring-swallow). p_real ~= 0.
- Conclusao: best-of-N cru inviavel ainda (0xN=0); ast.parse exige hardening (>=N stmts reais).
- Insight estrategico: oraculo deve atuar na saida do decoder VQ (128 BPE tokens c/ softmax 8192)
  via grammar-constrained decode, nao nos 641 codigos. Treino segue (CE caindo), proxima
  alavanca = constrained decode ou repair head quando houver material.


## CONSTRAINED DECODE RUN-041 2026-11-16 (gen_code2 @ ep10 CE 1.7108)

scripts/_constr_decode.py — rerank top-16 do decoder VQ por posicao c/ codeop prefixo-viavel.
- gen(32,t0.6,k24): argmax 0/32 parse → constrained 4/32 (12.5%) — lift real MAS
  hits sao degenerados (1 stmt + comments/strings; oraculo exploravel por #/""").
- forced_avg ~80-90/128: top-16 raramente contem continuacao viavel — massa de
  probabilidade nao sustenta programas validos ainda; constraint nao repara
  trajetoria descarrilhada.
- gold rows: 0/32 c/ e s/ constraint — slices mid-file nunca sao prefixo valido
  (esperado; oraculo so faz sentido em texto gerado do zero).
- Veredito: constrained decode sozinho NAO fecha o gap; end-score ranking
  (best-of-N com score = stmts reais + %chars em codigo) e proximo degrau, ou
  repair head. Oraculo incremental precisaria banir entrada em comment/string.


## TREINO CONTINUO 2026-11-16/17 — run estendido alem das 15 epocas

Trajetoria CE (epocas globais, gen_code2.pt):
5.165→4.504→4.184→3.895→3.609→3.244→2.878→2.516→2.188→1.917→1.711→
1.546→1.397→1.267→1.153→1.058→0.984 (ep16). Deltas recentes: -0.15/-0.13/-0.11/-0.074.
Run segue vivo (pid 200686, EPOCHS>12) — CE cruzou 1.0 sem plato.


## RE-MEDICAO constrained decode @ CE 0.9844 (ep16)
gen(32,t0.6,k24): argmax parse 1/32(real 0) -> constrained parse 8/32 real 7/32.
vs ep10(CE1.71): parse 4->8, forced_avg 80.5->52.3. Top-16 sustenta continuacao
viavel em ~48% posicoes (antes ~37%). Hits ainda degenerados (1 stmt + filler)
mas logica real emerge (if operator==...return). Tendencia composta: CE descendo
=> constraint melhora. Proxima alavanca segue: oraculo end-score / repair head.


## RUN-042 RELAUNCH 19k ROWS 2026-11-17

Overfit confirmado (held-out fine acc 11.8%@CE2.5 → 9.3%@CE0.93) → run anterior
matado (pid 200686). Nuevo run: pid 844317, N_ROWS=19000 (era 8000, 2.4x datos),
warm-start INIT=gen_code2.pt (CE 0.93), salida GEN_OUT=gen_code3.pt,
log train_code4.log, EPOCHS=40, sin COMPILE (dynamo InvalidCxxCompiler sin g++).
Backup: gen_code2.ep17-ce0.93.bak.pt.
NUEVO HOLDOUT: rows 19000:20000 (OFFSET=19000 para _gen_acc.py). Las rows
8000:8064 ahora son de entrenamiento — no usar OFFSET=8000 para held-out.
Baseline free-running pre-train del ckpt heredado: 2/32 parse+compile (temp0.9).


## RUN-043 PLANO B PROBE — transformer BPE direto vs VQ-641 2026-11-17/18

Hipotese: a representacao 641-codigos VQ e o gargalo de geracao.
Experimento: GPT decoder-only 55.21M params (dim768/L6/H12, ctx128, vocab8192)
treinado DIRETO nas mesmas rows BPE (19000 treino, 19000:20000 held-out),
3h wall CPU, THREADS=4 nice+15 em lcm-infer --network none
(launch: artifacts/ncm_code/bpe_probe_launch.txt; script scripts/_bpe_probe.py;
cache rows: bpe_rows.npy; ckpt: gen_bpe_probe.pt).

Resultado (bpe_probe_metrics_eval.json, held-out completo 1000 rows):
  train CE 9.01→2.54 em 1406 steps (~1.2 epochs)
  heldout_ce 3.26 · heldout_top1 39.34%
  argmax  0/32 parse · 0 real · filler_frac 0.0
  sampled 0/32 parse · 0 real · filler_frac 0.101

VEREDITO: VQ-641 NAO confirmado como gargalo. BPE direto com budget igual/menor
tambem gera 0/32 — consistente com a hipotese alternativa de que o bloqueio e a
FORMA DA TAREFA: rows sao fatias mid-file de 128 tok; geracao incondicional
produz continuacoes mid-file que nunca parseiam standalone (gold rows so
parseiam ~56% via canal). BPE direto segue viavel mas precisa: (a) profundidade
de treino igual, (b) eval alinhado a tarefa (prefix-completion), ou (c)
constrained decode nativo — que no BPE e mais simples (logits diretos).
Bonus do probe: heldout_ce/top1 limpos (metrica que o caminho VQ nao tem).

## RUN-044 FUNC-UNITS PROBE — task-aligned training (fronteira def/class) 2026-11-18

**Pergunta**: treinar em unidades standalone por fronteira de função/classe torna a geração livre parseável?

**Setup**: extrator `scripts/_func_units.py` — blocos `def|class|async def` coluna-0 por indentação → `ast.parse` + ≥1 stmt real + BPE ≤127 tok. Funil: 82.9k blocos → 73.0k parse-ok → 25.8k fit-127 → **24.8k unidades** (`func_units.txt`, `func_units_stats.json`). Encode via caminho LESSONS+TAG → `codes_func.npy` (10000×641). Probe: warm-start `gen_code3.pt` → `gen_func1.pt`, `MAX_T=10800` (~3h), container `funcprobe` isolado (`--network none`, nice+15, THREADS=4). Patch no trainer: env `MAX_T` (corte por wall-clock) + env `TAG` (separa codes_func de codes_lessons).

**Resultados**:
- 232 steps (~0.37 época de 10k) · loss ruidoso 3.07→3.34→2.41→2.26→3.32→2.94→2.68→2.95→2.61 · média época 3.0744
- Eval interna (ast.parse leniente): pre 1/32 → post 0/32
- Eval endurecida (`_eval_func.py`, seed 7, N=32): argmax(top_k=1) 0/32 parse · sampled(t0.9/k32) 0/32 · real 0/32 · filler 0.00–0.019
- Geração **tem forma de função** (começa `def`, contém `assert`/`return`, argmax colapsa em 1 saída degenerada) — macro-estrutura aprendida, superfície corrompida

**Diagnóstico extra (canal)**: `_gold_func_channel.py` — gold func units encode→decode: **8/32 parse · text_sim 0.92**. O canal VQ→texto é fiel ~92% mas corrompe chars únicos que matam ast.parse. **Teto do canal ≈25% neste dado** — predição perfeita daria só ~8/32.

**Veredito**: task-alignment sozinho NÃO resolveu (0/32 nos dois regimes). Agora há DOIS gargalos empilhados medidos: (1) previsão de códigos finos (~10.5% top-1 held-out), (2) fidelidade do canal decode (~25% teto em func units). 

**Próxima alavanca proposta**: probe BPE direto sobre `func_units.txt` — elimina o canal (detokenize determinístico) E a falha de task design ao mesmo tempo; `_bpe_probe.py` + loader LESSONS. Alternativa: fine-tune do decoder VQ com objetivo parse-preserving.

**Trainer principal**: ncmtrain2 segue intacto (ep5 ~97%, pid 844317, ~750% CPU).

## RUN-045 BPE × FUNC-UNITS — PRIMEIRO RESULTADO POSITIVO 2026-11-18

**Pergunta**: BPE direto (sem canal) + unidades def/class alinhadas → geração parseável?

**Setup**: `_bpe_probe.py` + envs `UNITS_FILE/HOLDOUT/CKPT_NAME/METRICS_NAME`; 23000 train / 1800 holdout (disjuntos) de `func_units.txt`; 55.21M params; TRAIN_SECS=10800 (~1379 steps ≈ 1 época); seed 7.

**BUG ENCONTRADO (decode)**: `fast.decode()` do HF tokenizers devolvia glifos byte-level crus (`Ġ`, `Ċ`) — corrompia TODOS os textos de eval BPE anteriores. Fix: `id_to_token` + unmap `_BMAP` (int ordinals; str.translate não casa chaves str). Round-trip gold verificado.

**Resultados** (decode corrigido):
- held-out: **CE 1.30 · top-1 76.68%** (vs 39.34% do modelo em slices — task-align dobrou a previsão)
- geração prefix-seeded `def `: argmax 0/32 (colapso de modo + truncamento ctx), **sampled t0.6/k24: 18/32 parse · 18/32 real · filler 0.053**
- geração incondicional (semente token 0): 0/32 — start OOD (nenhuma row começa com id 0)
- Re-medição modelo RUN-043 (slices, decode corrigido): ainda 0/32 — sua falha era real, não artefato; held-out CE 3.26/top-1 39.34% confirmados

**Exemplos gerados**: `def rich_uniform(name: str) -> str: return string.encode("utf-8")`, `def _impl(self: ComplexTensor) -> ComplexTensor: return ComplexTensor(self)` — funções completas e válidas.

**Veredito**: POSITIVO — barra pré-registrada ≥6/32 superada 3× (18/32). Task-align + BPE direto produz código standalone parseável pela primeira vez. Gargalos remanescentes (mecânicos): start-token OOD, ctx-128 trunca funções, argmax colapsa (amostragem resolve).

**Próximos passos**: (1) BOS token ou prefix-seed como padrão de geração; (2) ctx 256 para funções maiores; (3) escalar (mais steps/capacidade) — sinal forte o suficiente para run sério.

## RUN-046 SCALE-UP BPE ctx256+BOS — NOVO RECORDE 2026-11-19

**Decisão estratégica**: usuário aprovou parar `ncmtrain2` (VQ, 50h, ep8) e alocar CPU dedicada ao caminho vencedor. `gen_code3.pt` ep8 preservado (479MB). Justificativa do cálculo: teto VQ estrutural ~25% parse (canal) + fine-pred ~11% held-out plano há 3 épocas → VQ nunca alcançaria o BPE já medido.

**Setup**: `_func_units.py` env `MAX_TOK=255`/`OUT_NAME` → `func_units256.txt` (38,613 unidades, +56%, tok_mean 108.5, máx 255). `_bpe_probe.py`: `CTX` env + layout `[0]+toks[:CTX-2]+[0]+pad(1)` (token 0 = BOS+EOS in-distribution) + `INIT` warm-start + cache `bpe_<stem>_ctx<N>_rows.npy`. Container `bpe256`: 36000 train/2613 held disjuntos, 55.31M, TRAIN_SECS=21600, THREADS=8, nice+10, --network none.

**Resultado**: 4266 steps (~1.9 épocas), train CE 2.84→0.78.
- held-out: **CE 0.9162 · top-1 82.43%** (RUN-045: 1.2971/76.68%)
- **sampled `[0]+def `: 26/32 parse · 26/32 real** ← novo recorde (RUN-045: 18/32); threshold ≥24 = scale-up confirmado
- sampled `[0]` só: 16/32 · argmax: 0/32 (saída única com 1 linha corrompida `+ 1] + 1)[0]` — overfit tardio desvia argmax p/ função memorizada truncável)
- mid-run ~step 1750: argmax 32/32 genuíno + sampled 15/32 + heldout CE 1.0874 — eval mid-run funciona via EVAL_ONLY em segundo container nice+19
- Taxonomia dos 6 fails finais: ~3 truncamento (>255 tok), ~3 corrupção de brackets

**BUG DE EVAL (detectado+corrigido)**: `_eval_bpe_prefix.py` com `BOS=1` incluía BOS nos ids decodificados → `decode()` truncava na pos 0 → strings vazias → parse 32/real 0 (assinatura que entregou). Fix: strip `r[1:] if r[0]==0` antes do decode. Scope: só esse script+flag — evals in-run (seed removido por `idx[:,1:]`), RUN-045 e toda a linha VQ não foram afetados.

**Scripts configuráveis agora**: `_func_units.py` (MAX_TOK/OUT_NAME), `_bpe_probe.py` (CTX/INIT/BOS layout/cache por ctx), `_eval_bpe_prefix.py` (CKPT_NAME/PREFIX/BOS, outputs por ckpt-stem).

**Estado**: CPU livre. `gen_bpe_func256.pt` = melhor gerador de código do projeto (55M, 26/32=81% parse sampled). Próximas alavancas ordenadas: (1) leg-2 via `INIT=gen_bpe_func256.pt` mais ~6h (loss ainda cedia em 0.77); (2) filler 0.262 alto no regime prefix — medir/curar docstring-weight; (3) atacar argmax (top_p ou beam curto); (4) mais dados (corpus tem ~165 shards, só 82.9k blocos extraídos).

---

## RUN-047 (em curso) — leg-2 + deep eval + TEACHER DISTILLATION PROBE

**Mudança estratégica**: usuário propôs "tornar o modelo bom aprendedor/comunicador primeiro, depois injetar conhecimento de professor forte e congelar" — rejeitou Qwen-30B como teacher (rouba recursos); o assistente virou o teacher via (a) unidades elite manuscritas + (b) sintetizador de famílias de padrões (tudo ast.parse-validado por construção).

**Setup paralelo** (todos CPU-only, rootless, --network none):
- `bpe256l2`: leg-2 warm-start de `gen_bpe_func256.leg1.pt`, +6h (nice+15)
- `teachdeep`: `_eval_deep256.py` no ckpt leg-1 (deep metrics em `gen_bpe_func256.leg1_deep_metrics.json`)
- `bpeteacher`/`teachx`: probe distillation — `teacher_all.txt` (63 elite + 4000 synth v1, 16 famílias), warm-start leg-1, 223 steps/1h → `gen_bpe_teacher.pt`

**Deep eval leg-1 (baseline)**: argmax 'def ' 0/8 (atrator corrompido) · 'class '/'def test_'/'def _' 8/8 · 'async def ' 0/8 · topk24 26/32 uniq26 filler .262 (doc_mean .246 — filler é quase tudo docstring) · topp09 23/32 filler .184.

**TEACHER PROBE — resultados (gen_bpe_teacher.pt, 223 steps, loss 0.07)**:
- Teacher-dist heldout: CE 0.0616 top-1 97.9% (memorização completa — synth é regular)
- **BOS-only: argmax 32/32 real · sampled 32/32 real filler~0.01** — confiabilidade sintática perfeita no modo default (leg-1: 0/32 e 16/32)
- argmax 'def ' 8/8 — atrator reparado: emite `def rich_user(items): return [x for x in items if x is not None]` (vs near-miss corrompido do leg-1)
- **FORGETTING MEDIDO (cross-eval em heldout REAL, mesmas 2613 rows): CE 0.9162→1.3543 (+48%), top-1 82.4→76.5%** — piorou em prever código real diverso
- **Cobertura estreita**: argmax 'class ' regrediu 8/8→0/8 e 'async def ' segue 0/8 — synth v1 (16 famílias) quase não tinha class/async; argmax não foi "consertado", foi **redistribuído** pros padrões cobertos
- sampled prefix-shifted caiu 26→21 uniq; textos fora do atrator mostram degeneração por repetição (`data.data.data...`)

**Bugs achados no caminho**: (1) `fam_accumulate` v1 falhava 100% por indentação — corpus v1 tinha só 15 famílias efetivas; (2) classes manuscritas com `\n\n` interno fatiavam no separador (63→71 fragmentos, 8 inválidos) — corrigido.

**Synth v2 gerado** (`teacher_synth_v2.txt`, 4000 units, 0 parse_fail, seed 11): +10 famílias cobrindo os buracos medidos — async(188)/class-methods(299)/decorated(197)/with/while/comprehension/lambda/imports/typed/docstring. `teacher_all_v2.txt` = 63 elite + 4000 v2 = 4063 units, 100% parse.

**Probe em curso `bpemix`**: mix replay — 26562 train (4062 teacher v1 + 22500 real, ~15%) + **mesmas 2613 heldout reais** → responde se dá pra manter BOS-32/32 + argmax limpo SEM o +0.44 CE de forgetting. CKPT `gen_bpe_mix.pt`, 1h, nice+19.

**Scripts**: `_teacher_gen.py` (26 famílias, SEED/N_OUT/OUT_NAME envs); `mix_25_75.txt` gerado por shuffle seed 7.

**Interpretação**: distillation MOVEU o ponteiro (1h fez o que 12h de corpus não fizeram) — direção validada, dose errada. Hipótese ativa: replay mix preserva os ganhos sintáticos sem forgetting. Decisão pendente do resultado `bpemix` + leg-2 final.

**LEG-2 VÝSLEDEK** (`gen_bpe_func256.pt` → snapshot `gen_bpe_func256.leg2.pt`): 3219 steps, train 0.46; **heldout CE 0.9345 (+0.018 vs leg-1 — plató potvrzen), sampled [0] 18/32, argmax 0/32**. Corpus route convergovala — další epochy = memorizace.

**MIX-15% PROBE** (`gen_bpe_mix.pt`, 208 steps): CE real 0.9392 (≈žádný forgetting) + argmax [0] 32/32, ALE sampled [0] 10/32 a topk24 18/32 — nízká dávka perturbuje stochastické cesty bez přenosu čistých návyků. Závěr: mix raso = worst of both.

**CURRICULUM B — VÍTĚZ** (`gen_bpe_curr.pt` = leg-2 → teacher_all_v2, 323 steps @ LR 1e-4, 25min):
- BOS-only: argmax 32/32 + sampled 32/32 · filler~0.02
- **topk24 'def ': 32/32 parse · uniq 31 — recorde** · topp09: 32/32 uniq 31
- argmax prefixy: 'def ' 8/8 ✓ · 'async def ' 8/8 ✓ (první model vůbec) · 'def _' 8/8 · 'class ' 0/8 · 'def test_' 0/8 (burky přesunuté, ne odstraněné)
- Forgetting: heldout CE real 1.044 (+0.11 vs leg-2) — malý drift vs +0.44 u FT puro
- Vzorky: čisté idiomatické jednořádkové funkce (setdefault, {**a,**b}, docstringy)

**Závěr**: distillation na konvergovaný base jako finální fáze = vítězná receptura. `gen_bpe_curr.pt` = aktuálně nejlepší generátor. Zbývající díry: argmax 'class '/'def test_' — pokrytí class-templates v synth rodinách stále slabé.

**Checkpointy**: leg1.pt (26/32 prefix baseline) · leg2.pt (corpus converged) · gen_bpe_teacher.pt (FT puro) · gen_bpe_mix.pt (mix) · **gen_bpe_curr.pt (produkční kandidát)**.

**Blind review RUN-047** (`.agentic/reviews/RUN-047-blind-review.md`): 0 blockers / 4 majors. Remediação enfileirada p/ próximo run: (1) decode() deve mapear cp→byte corretamente p/ todo o range (incl. 289-322/323 faltantes) + latin-1→utf8 reassembly — parse counts atuais são CONSERVADORES (mojibake→fail); (2) mascarar pads no heldout CE/top1; (3) cache key deve incluir hash/mtime do units file; (4) stats write só com sufixo .txt validado.

---

## PROBES pós-RUN-047 (linha intent/v3, sessão paralela — sem run formal)
## NOTA: o RUN-048 formal do agentloop pertence a outra demanda
## (painel renderizador NCM + dump LoRA, DONE em paralelo). Não renumerar.

**Pergunta científica**: canal `# intenção` → código funciona? E logits de
professor externo (Qwen) transferem entre arquiteturas/vocabs diferentes?

**curr v3** (`gen_bpe_currv3.pt` = leg-2 → teacher_all_v3, 328 steps @ LR
1e-4): BOS argmax 32/32 + sampled 32/32 mantidos; teacher-dist CE 0.066.
`_teacher_gen_v3.py` envolve units com `# intenção` PT/EN (~80%) + 20% bare.

**Eval de intenção (`_eval_intent.py`, métrica nova)**: 12 intenções × 8
amostras sampled t0.6 — **93/96 parse · 59/96 (61%) relevante** (heurística
lexical). Aderência segue cobertura: pi/freq/soma/dicts/palindrome/async =
7-8/8; arquivo/busca-binária/média/ordena = 0-1/8. **Canal confirmado;
gargalo é cobertura de domínio, não formato.**

**Probe Qwen** (`llama-server` local :8085, ctx 4096, t8):
- `Qwen3-1.7B-Q8`: 19/20 (95%) parse em intent→code; token_logprob ≈ 0 —
  logprob filtro descartado (sinal plano)
- `Qwen3-Coder-30B-A3B-IQ2_M` (~22 tok/s): 17/20 (85%); falhas = prosa
  vazada/corte/docstring residual — todas filtráveis por ast.parse
- **Logit-KD crua inviável**: vocab 8.192 vs ~150k + fronteiras
  desalinhadas. Alternativa correta documentada: ULD (arXiv:2402.12030) /
  DSKD (arXiv:2404.13715) — fora de escopo agora.
- Valor real medido: Qwen = **gerador de diversidade em escala** sob
  curadoria (eu formato/filtro, ele produz matéria-prima).

**Corpus Qwen gerado**: `qwen_units.txt` — 139/148 válidos (94%), 16min,
6.4s/unit médio. CAVEAT: docstrings + comentários "example usage" vazaram
do filtro (só parse checado) — strip necessário antes do curriculum v4.

**Docs**: `RESEARCH_LOG.md` criado — narrativa completa + referências
anotadas (arXiv IDs) + reprodução-do-zero + impossibilidades resolvidas.

**Pendente**: curriculum v4 (elite + synth v3 + qwen_units); remediações
blind-review; buracos `class `/`def test_`; server Qwen 30B ainda no ar
(~11GB RSS, matar com `pkill -x llama-server` se não for usar).
