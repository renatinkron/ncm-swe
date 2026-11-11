# Architecture Evaluation — lever DistillSpec-KL + top-k verify

## State

- `train_token_head.py`: variants {feat, codes}, CE loss contra
  corpus truth. Mudança: novo modo de alvo `--target ar`
  (soft-CE contra logits do backbone congelado na mesma posição).
  Estado novo: nenhum — só tensor alvo diferente; checkpointing
  idêntico (out separado `token_head_ar_qwen05` p/ não clobberar
  o ckpt truth anterior).
- `spec_pyramid.py`: verify greedy posição-a-posição sobre drafts
  top-1. Mudança: draft emite top-k por offset; verify aceita a
  PRIMEIRA posição cujo argmax AR ∈ top-k do draft (greedy
  k-candidate) → preserva equal_ar (token final = argmax do alvo).

## Flow

Treino: forward backbone (feat) → heads RVQ (códigos previstos) →
token-head → KL(student || AR_probs(T=1)). Mesmas posições/dados.
Inferência: draft K offsets × top-k → verify batched (K×k ou
argmax∈topk por posição — mais barato: comparação de conjunto, não
candidatos expandidos).

## Computation

- Soft-CE contra 152k logits: custo idêntico ao CE atual.
- Verify com top-k: k=8 → checar `argmax ∈ topk` por posição é
  O(K·k) comparações — trivial; não expande forwards.
- Risco de custo: nenhum além do treino ~1h.

## Failure

- Se soft-KL falhar em superar hard-CE: dataset de logits AR é
  autoconsistente por definição (alvo gerado pelo mesmo backbone
  no mesmo contexto) → falha indicaria problema de otimização, não
  de conceito. Mitigação: logar também a1 contra argmax (métrica
  já existente — vira a métrica primária).
- Se top-k ainda der accept=0 com a1>0: gap de calibração
  (logits codes mal-ordenados no top-k). Mitigação: reportar
  rank do argmax AR sob a distribuição do head (mesmo diagnóstico
  da RUN-009, agora sobre a head).

## Trust boundary

Sem mudança: tudo roda local, código próprio, modelos locais.

## Trade-offs

- Alvo AR em vez de truth: o head aprende "o que o AR diria"
  (pode repetir erros do AR) — aceitável: o objetivo é acceptance,
  não qualidade superior; truth-training já medido e estagnado.
- Top-k relaxa "o draft propõe 1 token" para "1 conjunto" — o
  NCM puro vira "concepto propõe shortlist" (coerente com a tese
  de coarse→fine: o conceito delimita, o verificador escolhe).

## Validation criteria

- a1 codes > 0 (destravar do zero) — métrica primária.
- accept_per_offset > 0 em algum offset com equal_ar=True.
- Spec tok/s deve subir vs 1,57 (RUN-011) se accept>0.
