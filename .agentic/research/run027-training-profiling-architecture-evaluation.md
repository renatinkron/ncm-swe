# RUN-027 architecture evaluation — otimização de performance treino/inferência

## Decisões e trade-offs

### D1 — Onde investir: treino vs inferência
Estado: treino compute-bound (~67% GEMM MKL @ ~84% pico Zen2); inferência
overhead/memory-bound (~190–450ms/passe vs ~10–30ms roofline).
**Decisão**: treino recebe só a alavanca barata (trunk-compile opt-in);
inferência é onde mora o ROI de engenharia (5–15× plausível vs ~1,15×).

### D2 — torch.compile trunk-only no trainer
Fluxo: flag `--compile-trunk`; wrapper `trunk(h)` sobre `model.blocks`
compilado uma vez por processo (shapes estáticas: B, T fixos).
- Trust boundary: inductor gera código C++ — exige imagem com g++
  (`lcm-infer-gcc`). Fallback silencioso para eager se compile falhar.
- Falha conhecida: forward inteiro quebra no head loop (stride assert,
  torch 2.5.1) — recorte trunk-only medido −14%.
- Validação: mesma loss em N steps seed-fixado compiled vs eager
  (tolerância fp32 de reassociação de fusão — bit-exact não é exigível;
  aceitar |Δloss| relativo < 1e-5 e trajetória idêntica).
- Risco: warmup ~2–4min/run — amortiza em runs >1h; não usar em sanity tests.

### D3 — Prefix/KV caching no decode intra (futuro braço de inferência)
Estado: `Block.forward` já aceita `past_kv` e retorna `new_kv` (API pronta).
Fluxo hoje: `_forward_partial(codes, cur, decided)` recomputa q0+todos os
chunks a cada passe intra (até 4×/nível com intra_passes=32) e a cada
rodada de refine.
Problema: chunk i depende de z_hat dos níveis ≤ i — prefixo até chunk k−1
é CONGELADO enquanto decodamos nível k. Cacheável por nível:
- cache[layer] = (K,V) do prefixo [q0 + chunks 0..k−1]; recomputa-se só o
  chunk corrente (q0-attend: chunks novos veem q0 via KV — posição 0 inclusa).
- Invalidação: ao transitar de nível, append do novo chunk; em refine, o
  chunk corrente muda (codes remascarados) → invalidar só a cauda do chunk
  corrente, manter prefixo.
- Risco de correção: RoPE é absoluto — K cacheadas já trazem rotação
  aplicada; posições não mudam → cache válido. Atenção block-diagonal é
  estática por posição → máscara fatiável por chunk.
- Validação: decode com cache deve reproduzir idx idêntico (mesma seed)
  vs sem cache — assert em eval de fumaça.

### D4 — Engine C/Rust dedicada
Trade-off: custo de port alto (atenção block-diagonal + VQ hierarchy +
mask_emb + refinement são custom; ggml/candle não têm esse grafo pronto)
vs ganho plausível 2–4× no decode já coberto em parte por D3+batching.
**Decisão**: adiar; revisitar se D3+batching insuficiente. O caso kimi-k3-in-c
mostra que engine C paga quando o gargalo é I/O/streaming — aqui é dispatch
+ recompute, resolvível dentro do PyTorch primeiro.

### D5 — Batching de eval (n amostras em 1 geração)
`generate(batch_size=n)` já existe — eval sequencial é escolha do sampler,
não limitação do modelo. Trocar n=8 sequencial → batch 8: ~8× no eval,
zero risco científico (mesma receita, seeds por amostra — verificar
reprodutibilidade de RNG batelado vs sequencial antes de trocar o protocolo;
se divergir, manter protocolo e aceitar custo).

## Critérios de validação persistidos
- compiled: |Δloss| < 1e-5 rel. em 20 steps seed-fixado; s/step −10%±3.
- prefix-cache: idx idêntico ao decode sem cache (seed 0, n=4);
  s/amostra intra ≥ −40% com refine=1.
- qualquer otimização de treino: não altera métricas científicas
  (dup/gap/CE dentro do ruído n=8 documentado).
- nada muda o protocolo oficial de amostragem dos resultados publicados.
