# RUN-012 Blind Review

Changed files: `scripts/train_token_head.py` (+--target ar, +--temperature,
+soft-CE branch), `scripts/spec_pyramid.py` (+--topk, draft_token returns
tuple, frontier rescue verify), `scripts/eval_token_head.py` (novo),
+ artifacts JSONs.

## Revisão a frio

### spec_pyramid.py — verify frontier

- Semântica verificada: aceita posição se `ar[j] ∈ cand_set`; emite
  `ar[:m]` (sem bonus) e força `feat=None` — correto porque o hidden
  seguinte foi computado com o top1 do draft (contexto stale após
  divergência). `equal_ar` preservado estruturalmente: todo token
  emitido é um argmax do backbone. Confirmado empiricamente: 3 benchs,
  text_equal_ar=True.
- `draft_token` retorna `int(idx[0,0])` como top1 — correto após fix
  (a versão inicial `int(vals.argmax())` retornava literalmente 0 —
  foi capturada no próprio desenvolvimento, pré-bench).
- `made += len(accepted) if frontier else m + 1` e o trim final
  `ids[:, :-(made-n)]` consistentes: made conta tokens emitidos.
- `accepted_hist[m]` no frontier registra m (não m+1) — coerente com
  a semântica "drafts aceitos".
- `accept_per_offset[j]` para j>0 tem denominador pequeno enviesado
  (só conta rounds com top1 exato em pos0) — NÃO é bug, é a semântica
  do frontier-greedy; reportado corretamente via accepted_hist no
  relatório. Poderia ter um comentário no código explicando — menor.

### train_token_head.py — alvo AR

- `ar_logits = feat @ E.t()` reutilizado (antes só argmax) — sem custo
  extra. `soft_tgt = log_softmax(ar_logits/T)`; loss =
  `-(exp(soft_tgt)·log_softmax(logits)).sum.mean()` = CE(p_ar,q) —
  forma correta de soft-CE (gradiente ≡ KL a menos de constante).
- Posição do alvo: mesma do tgt/ar_tok (row[-1] via feat de inp) —
  sem deslocamento. Conforme contrato.
- Retrocompat: default truth → ramo else intocado.
- Warning preexistente `float(loss)` com grad — cosmético, já
  documentado na RUN-011.

### eval_token_head.py

- Split idêntico ao trainer (últimos 5%, seq=64) — mesma fatia.
- Rank: `order==ar_tok` → coluna do nonzero = rank 0-based +1 —
  correto (1 hit por linha).
- Persistido no repo (fecha a ressalva do blind review RUN-011).

## Findings

- Nenhum major/minor funcional.
- Observação: `accept_per_offset[1..3]` pode induzir leitores ao
  erro (denominador minúsculo); report já usa accepted_hist.
- Frontier-only é honestamente documentado como não-speedup
  (trade-off forwards×acceptance reportado).

verdict: clean
