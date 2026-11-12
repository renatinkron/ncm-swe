# kimi-k3-in-c — extração (11 nov 2026)

Repo: github.com/FareedKhan-dev/kimi-k3-in-c (11,9k★, Apache-2.0)
Claim: Kimi K3 (2,78T params, MoE 1,56TB ckpt) em C99 puro, 176KB de engine,
8,24GB RAM num CPU — byte-identical output em qualquer budget 8→224GB.

## Técnicas extraíveis (por relevância)

1. **MoE expert streaming** — 896 experts/layer, só 16/token ativos → nunca
   residentes. Cada expert = 17,55MB MXFP4. Cache LRU de 3 estados
   (EMPTY/INFLIGHT/pinned) com protocolo de 3 fases: reserva serial → leitura
   paralela em batch ordenada por offset de disco → publica só o que chegou.
   Fallback de leitura coalescida (1 read de 17,55MB) vs 6 tensores.
   → DIRETO pro Colibri (Qwen3-Coder-30B-A3B é MoE) e pra conversão MoE→NCM.

2. **MXFP4 in-place matmul** — nunca dequantiza: nibble E2M1 via LUT de 16
   entradas + escala E8M0 por grupo de 32 (0,53 bytes/peso). Lição de
   validação: "nibble order trocada tem as mesmas estatísticas" — checagem
   por distribuição não detecta; só posição detecta. (Padrão aplicável aos
   nossos testes de tokenizer/codes.)

3. **Trunk pinned-PREFIX + ring slot** — LRU sobre scan cíclico = 0% hit
   (caso patológico: quando camada 0 volta, é sempre a LRU). Prefix-pin =
   hit determinístico N/93. Contra-exemplo canônico de "óbvio errado".
   → Aplicável a qualquer streaming de camadas sequenciais (inclusive o
   nosso forward NCM de 4 camadas — trivial, mas o princípio vale pro trunk).

4. **KDA (Kimi Delta Attention)** — estado recorrente fixo 128×128/head,
   626MB total p/ 93 camadas em QUALQUER contexto. Linear attention não
   cresce com contexto. + MLA (1 latent vs 96 KV heads).
   → Candidato arquitetural se um dia portarmos NCM p/ modelo maior.

5. **I/O engineering** — O_DIRECT > buffered na máquina deles (medido!);
   arenas em hugepages 2MB (evita ~54M page-ops/token); contabiliza TUDO
   antes de alocar e RECUSA start se shortfall >5%.

6. **Metodologia (ouro p/ nosso rigor)**:
   - cgroup MemoryMax + MemorySwapMax=0 (sem cap de swap, mede swap bw)
   - cache sizing via trace replay (routing não depende do cache → roda 1×,
     replay em todas capacidades; Belady como teto)
   - A/B num MESMO binário via env var, não dois builds
   - "a speedup that changes tokens is worthless" — byte-identical como gate
   - NÃO reivindica ganhos dentro do ruído (melhora de 5,4% reportada como
     "mechanism sound, measurement does not establish") — igual à nossa
     honestidade n=8

## Leitura honesta p/ nosso programa

- 8GB → 32,7 s/token: viabilidade, não usabilidade. Nem 128GB (5,6s/tok)
  chega a interativo. **Reforça nossa tese**: assistente de código local
  interativo provavelmente NÃO virá de "modelo gigante + streaming" —
  o caminho NCM (modelo pequeno dedicado + decode paralelo) permanece a
  aposta mais plausível p/ latência.
- "Storage is the whole game": 41-61% do wall-clock é espera de disco em
  TODOS os budgets. Pra nós: o speedup NCM vem de passes sequenciais, mas
  sob baixa RAM o gargalo migra p/ I/O — recontextualiza onde a
  paralelização ajuda.
- MoE streaming é o upgrade path direto do Colibri se voltarmos àquele
  runtime (30B-A3B: experts streamáveis + shared resident).

## Referência p/ paper

Candidato a citação em trabalhos relacionados (engenharia de inferência
local extrema) — blog técnico não-revisado, mesmo estatuto do WPS:
evidência qualitativa convergente, não referência quantitativa primária.
