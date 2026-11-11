# Architecture evaluation — publicação do estudo NCM como trabalho acadêmico

## State
Artefatos-fonte: `artifacts/report.md` (607 linhas, RUN-001→020), métricas
`*/metrics.jsonl`, `bench_*.json`/`eval_*.json`, run states `.agentic/state/runs/`,
RAG papers em `.agentic/research/rag/papers/`, colibri `SESSION_HANDOFF.md`.
Produto: repo público novo + documento ABNT PT-BR/EN-US.

## Flow
report.md → extração de claims/numbers → redação ABNT (PT) → tradução EN →
2 avaliações independentes (subagentes frios) → 2 revisões gramaticais ABNT
independentes → correções → README/LICENSE/repo público → push.

## Computation
Números centrais a preservar literalmente (anti-alucinação): speedups
×1,05–1,34 (60M), ×2,27 (471M), ×2,82/×3,22/×3,21 (TinyStories), held_loss
7,55→5,82 pós-fix, held_l11=2,68 < oficial 2,75, l0≈8,2≈uniforme(8,32),
k6=96,9% ablação, 0 accepts→0,58@k=64 spec-decode, compute ratio ~2000×.

## Failure modes
- Alucinação de citação: toda referência deve sair de arquivo local ou web já
  verificada (lista no research brief); proibir DOI/URL inventada.
- Vazamento de dados pessoais: scan por `/home/kron`, `kron`, hostname, paths
  do colibri privados, IPs antes do push; checkpoints >100MB não vão ao repo.
- Falso posicionamento: deixar explícito que é relato de replicação parcial
  + análise de falhas, não validação da receita oficial completa.
- "Agente independente sem histórico": usar run_subagent sem repassar
  contexto além do arquivo do paper e a tarefa.

## Trust boundaries
Repo público = fronteira irreversível: qualquer segredo/PII commitado vaza.
Pré-push: grep de sanitização + git log review. Push só após veredito limpo
dos 4 revisores independentes e aprovação tácita do fluxo (usuário já pediu
publicar — não reconfirmar, mas relatar URLs finais).

## Validation criteria
- ABNT: estrutura NBR 14724:2024, refs NBR 6023:2025, citações NBR 10520:2023,
  resumo/abstract NBR 6028:2021, seções NBR 6024:2012.
- Toda referência rastreável; nenhum número no texto sem fonte nos artifacts.
- LICENSE presente; README com resumo + como citar; histórico git com datas.
