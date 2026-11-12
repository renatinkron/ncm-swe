# RUN-031 blind review — monitoramento v5-ext

Escopo frio: apenas artefatos de estado/log deste run — nenhum código
novo (watcher/eval scripts revisados em RUN-030; treino usa trainer e
flags já validados).

## Verificações

- Nenhum arquivo de código alterado neste run.
- Scripts em uso (`watch_v5ext.sh`, `eval_ckpt_v5ext.sh`) revisados no
  blind review do RUN-030 — sem drift desde então.
- Critério de leitura do report (Δ<0,1 nat inconclusivo em n=8) é
  consistente com a correção estatística do RUN-030 — correto exigir
  confirmação n=32 para o veredito final.
- Log factual sem extrapolação.

0 blockers, 0 majors, 0 nits. Nada a corrigir.
