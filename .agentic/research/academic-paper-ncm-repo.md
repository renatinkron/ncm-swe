# Trabalho acadêmico NCM + repositório público (ABNT, bilíngue, revisão multiagente)

## Context
O usuário quer compartilhar o estudo NCM com um amigo doutorando que tem acesso a
clusters. O valor do material: replicação independente parcial do NCM em CPU
comum, speedup hierárquico medido crescendo com escala (×1,14→×2,27→×2,82-3,22),
descoberta de um bug de init que invalidava vereditos, e análise sistemática de
falhas em 20 runs — exatamente o tipo de "o que não funciona e por quê" que
economiza tempo de laboratório. O estudo colibri (Qwen3-Coder-30B MoE em 16GB,
~0,6-0,9 tok/s) motivou a busca por paradigma de geração alternativo.

Resposta à pergunta do usuário ("faz sentido compartilhar?"): **sim** — resultado
negativo bem documentado + bug real + hipótese testável em cluster é material
científico legítimo, formato de relato técnico/replicação.

## Constraints
- Sem remote git no repo; repo novo no GitHub do usuário (gh CLI).
- Histórico: mostrar timeline de pesquisa (commits/run dates out-30→nov-9) sem
  dados pessoais/identificáveis (paths /home/kron, nomes de máquina ok? "ThinkPad"
  é genérico; remover caminhos de home e dados do colibri privados).
- ABNT vigente: NBR 14724:2024 (apresentação), NBR 10520:2023 (citações),
  NBR 6023:2025 (referências — edição nova), NBR 6028:2021 (resumo),
  NBR 6024:2012 (numeração progressiva), NBR 6022:2018 (artigo).
- Licença "exclusivamente acadêmica/pesquisa": CC BY-NC 4.0 é o padrão de facto
  (Meta SONAR usa). Código separado sob MIT é convenção; aqui o artefato é o
  texto+scripts → CC BY-NC 4.0 cobre; alternativa "CC BY-NC-ND 4.0" mais
  restritiva mas impede traduções/derivações — rejeitada.
- Revisões exigidas: 2 avaliadores independentes sem contexto + 2 revisores
  gramaticais ABNT/PT-BR independentes (subagentes).
- Extensão: "não tão extenso" — formato artigo curto / extended abstract +
  relato técnico (padrão TinyStories/notes: introdução, método, resultados,
  limitações, próximos passos).

## Fontes coletadas (todas a citar)
- Kumbong et al. 2026, NCM arXiv:2610.14162 (paper central + repo + checkpoints)
- Eldan & Li 2023, TinyStories arXiv:2305.07759 (corpus)
- LCM/SOC — "Large Concept Models" Meta SONAR (comparativo última milha)
- Medusa Cai et al. 2024 arXiv:2401.10774; SpecInfer; EAGLE ICML'24;
  DistillSpec ICLR'24; RQ-Transformer CVPR'22; VAR; MaskGIT
- spec-survey Hu et al. arXiv:2502.19732; rvq-kv Kumar arXiv:2410.15704
- mcsmoe arXiv (Merge-Then-Compress); submoe arXiv:2506.23266;
  medlm Edit-Based Refinement arXiv:2605.09603
- MANA spec (conversa Gemini destilada — fonte secundária interna, citar como
  comunicação pessoal/relato interno)
- Qwen checkpoints HF; llama.cpp; Mimir-1.6B mirror
- colibri repo (SESSION_HANDOFF.md — motivação do estudo)

## Options
1. Artigo ABNT completo (resumo+abstract+intro+método+resultados+conclusão) —
   recomendado: encaixa "impecável mas não extenso".
2. Relato técnico informal — rejeitado (requisito ABNT).
3. Preprint estilo arXiv só em inglês — rejeitado (requisito PT-BR + EN).

## Recommendation
Repo novo `ncm-swe-study` (ou similar) público: `paper/` com `main.tex` (ou .md
formatado ABNT) PT-BR + versão EN-US; `README.md` sumarizando; `LICENSE`
CC BY-NC 4.0; `artifacts/` só com métricas/amostras-chave (sem checkpoints
479MB — usar releases ou omitir); histórico via commits datados e CHANGELOG
dos runs. Autor: Renato de Souza e Silva. Revisões por subagentes frios
(2 avaliações científicas + 2 revisões ABNT/ortografia) antes do push.

## Open Questions
- Formato do paper: LaTeX (padrão ABNT/ieee) vs Markdown formatado? LaTeX gera
  PDF impecável; verificar se há TeX no host/container.
- Nome exato do repo GitHub (propor `ncm-hierarchical-generation-study`).
- Incluir scripts de reprodução? Sim — é o valor pro lab com cluster.
