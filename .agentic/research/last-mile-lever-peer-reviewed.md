# Última milha NCM — alavanca mais promissora (evidência peer-reviewed)

## Context

RUN-007..011 mediram: drafts piramidais geram conceitos certos mas
accept=0 contra argmax AR. A pergunta da run: qual alavanca tem
suporte na literatura para destravar aceitação — sem achismo.

## Constraints

- CPU-only (4750U): intervenção deve custar ~1h de treino, não dias.
- Backbone/heads/codebook congelados; mudança só no head de token
  e no verificador.
- `equal_ar` é o contrato do spec-decode greedy.

## Options (evidência)

1. **DistillSpec (ICLR'24) + EAGLE (ICML'24) — treinar o draft
   contra a DISTRIBUIÇÃO do alvo, não o corpus.** DistillSpec
   prova que alinhamento draft↔target é o que determina acceptance,
   e que dados on-policy + divergência certa dão +10-45% sobre SD
   standard. EAGLE prova que autoregressão em espaço de feature é
   mais fácil que em token e decodifica via a própria head do alvo.
   → Implementação: trocar o alvo do `train_token_head.py` de
   hard-label (corpus) para KL/soft-CE contra os logits do backbone
   congelado. Custo zero de maquinaria: mesma loss loop, troca o
   tensor alvo. **Previsão da literatura: aceitação > 0.**

2. **SpecInfer (MLSys/SOSP'24) + Medusa (ICML'24) + Multi-Candidate
   SD — top-k/tree em vez de top-1.** Com draft 100× menor
   (68M→7B), acceptance greedy sobe 68%→85% com k=5; satura ~k=32.
   Medusa usa exatamente "heads sobre backbone congelado" (= nossa
   arquitetura) + typical-acceptance. → Implementação: `--score
   tokhead` emite top-k por posição; verify aceita se argmax AR ∈
   top-k (greedy k-candidate, mantém equal_ar pois o token emitido
   continua sendo o argmax do alvo). Custoso apenas em largura de
   batch no verify (k× posições — barato em 0.5B).

3. **RQ-Transformer (CVPR'22) / HQ-VAE (NeurIPS'22) — nossa pirâmide
   tem precedente publicado.** Depth-wise code-stack prediction
   conditioned on previous features = exatamente nossas LevelHeads;
   HQ-VAE amostra códigos finos em paralelo via independência
   condicional = nossos offsets. Legitimidade arquitetural: ok.

4. Soft-code conditioning (esperança de código) — refinamento de (1),
   não substituto; mantém-se como ablação posterior.

5. EAGLE-forma do head (prever feature 896d, decodificar via E.T) —
   alternativa estrutural interessante mas menos direta que KL;
   guardar para se (1)+(2) falharem.

## Recommendation

**Alavanca combinada (1)+(2) numa única run:** retreinar o
codes-head com alvo = distribuição do AR congelado (KL soft,
temperature 1 — "distilar o verificador no draft") e emitir top-k
candidatos por posição no verify. É a mudança mínima que ataca
diretamente a quantidade medida (P(argmax ∈ draft)), sustentada por
DistillSpec (treino contra alvo) + SpecInfer/Medusa (k candidatos).

Reuse check: nenhuma lib local pronta; Medusa/EAGLE existem como
repos mas assumem GPU+pipelines HF grandes demais para o nosso
escopo — reimplementar os 2 deltas (~60 linhas) é mais barato.
Cross-project: colibri não tem spec-decode; nossos scripts já
cobrem o esqueleto.

## Open Questions

- k ideal em CPU (começar k=8; literatura satura ~32).
- Se KL-soft superar hard-CE argmax puro (registrar ambos se
  barato: soft-CE T=1 como default).
- On-policy data (DistillSpec) — fase 2 se aceitação destravar.
