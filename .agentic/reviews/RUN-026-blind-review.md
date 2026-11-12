# Blind Review — RUN-026 (revisor frio, commit b55e70c)

## Veredicto: 0 blocker, 1 major, 6 minor, 10 nit

## Findings

| arquivo:linha | severidade | finding |
|---|---|---|
| generator.py:426-432 | **major** | **Confidence-scoring pass é auto-referente (self-leak).** Sob `all_dec`, o input da posição t já contém `emb(cur[t])` — `decided` soma o embedding do *próprio* código-alvo na posição que o prediz (`_intra_add`/`_forward_partial`: `torch.where(decided, ce, mask_emb)` na mesma posição do target). Logo `conf[t] = p(cur[t] \| contexto incluindo cur[t])` — a resposta está no input. Ademais, outputs em posições decided recebem zero sinal de treino (loss mascarada por `pending`), então a head nessas posições é não-treinada: `conf` mede "echo-ability"/lixo não-calibrado, não incerteza da predição. Se o modelo aprendeu a ecoar (capacidade livre), conf≈1 em todo lugar e `argsort` vira ordenação arbitrária; se não, o ranking reflete artefatos. O remask dos "menos confiantes" degenera para remask ~aleatório — a mecânica Mask-Predict está mecanicamente correta mas o sinal que ela ranqueia é vazio. Correto seria pontuar posições *enquanto pending* (ex.: 1 pass com máscara all-pending, ou reutilizar a prob atribuída no pass que a amostrou). Refinement ainda funciona como "random remask + repredict com contexto de pares" (tem valor), mas não faz o que o docstring anuncia. |
| eval_ckpt_v5.sh:43-46 | minor | `SUFFIX` codifica só `REFINE_ITERS` — `REFINE_FRAC`/`REFINE_FROM` diferentes colidem no mesmo `OUT` → "já avaliado" pula e serve eval obsoleto sob hiperparâmetros diferentes sem aviso. |
| generator.py:421 + sample_official.py | minor | `--refine-iters >0` silenciosamente ignorado em ckpt não-intra (branch `continue` antes do bloco); também no-op silencioso se `refine_from ≥ len(levels)` ou se usuário ler `refine_from` como tamanho-de-nível em vez de índice. JSON de saída não registra nenhum campo refine — impossível detectar pelo artefato (só o sufixo `_refN` no filename, e só iters). |
| generator.py:421-439 | minor | Refinement em ckpt `intra_mask='boundary'` é OOD: treino boundary só viu máscaras prefixo; `decided2` remascara subconjuntos arbitrários nunca vistos → predições fora-da-distribuição sem guarda nem warning. Para ckpts 'random'/'mix' é in-distribution. |
| generator.py:38-40 | minor | `intra_mask` não é validado no ctor: qualquer string ≠ 'boundary'/'mix' cai no branch `else` e silenciosamente se comporta como 'random' (o check `== 'mix'` falha e `dec` permanece `rank<k`). Typo em cfg de ckpt ou uso programático → troca de modo silenciosa. |
| sample_official.py:180-183 + eval_ckpt_v5.sh:66 | minor | Headline `ce_self_gen`/`ce_held_gt` em ckpts intra são medidos sob máscaras amostradas no modo `intra_mask` do ckpt → números dependentes da distribuição de máscara; comparação cross-arm (v4 flat × v5 mix) é apples-to-oranges. |
| generator.py:67-69 | minor | `mask_emb[0]` (nível 0) criado mas jamais usado — só `mask_emb[i+1]`/`[k_next]` para níveis preditos ≥1. Parâmetro morto (sem dano funcional). |
| generator.py:422 | nit | `n_mask = max(1, ...)`: `refine_frac=0` ainda remascara 1 posição/rodada (piso não documentado); e o default diverge — `generate()` tem `refine_frac=0.0`, sampler expõe `0.3`. |
| sample_official.py:122-123 | nit | `--intra-passes` negativo passa pelo `or` → `range(0,Lk,neg)` vazio → nível sai todo código-0 em silêncio; `intra_passes=0` gravado em cfg → `range` step 0 crash (misuse-only). |
| eval_ckpt_v5.sh:21 | nit | Label numérico/`[0-9]*` rotula pelo argumento, não pelo step real do ckpt.pt (nit herdado de v4, já DEFER em RUN-025). |
| eval_ckpt_v5.sh:27-32 | nit | Guard anti-torn-snapshot é bounded: após 5 re-stats copia de qualquer jeito (padrão v4 herdado; rasgo real só com >~10min de writes contínuos). |
| eval_ckpt_v5.sh:44 | nit | `[ "$REFINE_ITERS" != "0" ]` é string compare: `REFINE_ITERS="00"` → filename `_ref00` mas argparse `int→0` → sem refinement (cosmético). |
| generator.py:435-437 | nit | `new` amostra todas as Lk posições mas só pending são usadas — draws de RNG desperdiçados. |
| generator.py:428 | nit | `conf` via `softmax` completo + gather; `log_softmax`+gather equivalente e mais barato (trivial). |
| generator.py:318 | nit | `_forward_partial` não valida par `cur`/`decided` (cur sem decided → AttributeError em unsqueeze) — API interna. |
| train_gen_official.py:92 | nit | Mensagem "init aleatório" para `mask_emb` ausentes é enganosa — init é zeros (no-op intencional), não aleatório. |

## Trajectory

`RUN-026.events.jsonl` limpo: `task_plan` falhou o validador uma vez (faltava acceptance-criteria + enum de branch) e foi refeito — auto-correção normal, não falha oculta. `test` reportou nc=0 ~13s após `execute` (mudança dominada por código de modelo+scripts, plausível para contrato leve — sem evidência de expect silencioso). `run_state.tasks[0].status` ficou "pending" apesar do history registrar T001–T004 executados — campo de status dessincronizado, nit de processo.

## Patterns

3 findings convergem em **validação ausente → modo errado silencioso** (`intra_mask` não validado; refine-iters ignorado em não-intra; refinement OOD em ckpt boundary). O código prefere degradar quieto a falhar — padrão coerente com o estilo do projeto, mas concentra o risco real do diff.

## Verificados limpos pelo revisor

- `_sample_intra_masks`: truque perm/rank (argsort+scatter do inverso) verificado — `rank<k` seleciona exatamente k posições, subconjunto uniforme; 'mix' sorteia prefixo-vs-random por item por nível a 50/50 com `b,k ~ U{0..L}`; boundary mantém `randint(0,L+1)` por nível na coluna i correta.
- `pending` em `loss()`: `arange>=b` é complemento exato de `<b` (boundary); `~decided[i]` complemento exato (random/mix); normalização por `pending.sum()` batelada consistente com a semântica pré-existente.
- Refine loop *mecânica*: gather com idx (B,Lk,1) long ok; argsort ascendente = menor confiança primeiro ok; `scatter_(1, low, False)` remascara exatamente n_mask ok; `where(decided2, cur, new)` preserva decided e atualiza pending ok; `passes += 2`/iter e `last_passes` (+1 só quando l0 é model) ok; loop chunked cobre Lk inteiro antes do refinement (`cur`/`decided` completos).
- Backward-compat: `intra=False` → `_intra_add` nunca chamado, CE plain → idêntico; `refine_iters=0` → bloco inteiro skipado → idêntico; modo boundary mantém path por tensor `boundaries` (decided=None).
- Sem leak entre níveis: pending positions veem `mask_emb` (códigos próprios ocultos) tanto no decode chunked quanto no re-predict do refinement; posições futuras nunca vazam.
- `mask_emb` recebe gradiente — `_intra_add` roda *fora* do `no_grad` (vq congelado não acumula grad; where roteia grad só pro mask_emb nas posições pending).
- Corrupt × intra: `_corrupt_codes` atua em `idx_in` antes de `_intra_add` → códigos decided carregam corrupção — semântica consistente de teacher-corruption; targets permanecem limpos.
- Warm-start: assert tolera só `mask_emb.*` em missing+unexpected; mismatch real de arquitetura ainda aborta alto; `strict=True` no sampler é seguro porque cfg salvo inclui `intra*` nas duas escritas (ckpt.pt e final.pt).
- Scripts: OFFSET=9499, save@local499 → step 9998 = FIRST ✓; fórmula de BOUNDARY verificada em LAST∈{0, 9499, 9998, 10007, 10498} → {498(skip), 9498(skip), 9998, 9998, 10498}; a espera ≥120s do guard também cobre o lag do metrics-tail (próximo log aterra antes do NAME ser computado); filenames watcher↔eval consistentes (BOUNDARY vira label→NAME); `podman --network none` preservado; REFINE_FROM default 7 = índice → níveis L≥64 (5 níveis), escolha deliberada.

## Veredicto

Diff cirúrgico e majoritariamente correto: a generalização `_intra_add`/`forward`/`loss` para máscaras `decided` arbitrárias é limpa, a amostragem de máscaras está certa nos três modos, a compat-inversa está preservada, e os scripts são mirrors fiéis de v4 com aritmética de offset conferida. O único defeito substancial é algorítmico: o pass de confidence-scoring do refinement vaza o próprio código-alvo para dentro do input da posição avaliada (e aquela saída nunca foi treinada), tornando o ranking "menos confiante primeiro" arbitrário — o refinement roda, mas executa remask aleatório com re-prediction contextual, não Mask-Predict. Não é blocker (nada quebra; o modo ainda pode ajudar), mas o mecanismo central da feature não mede o que afirma medir — recomendo re-scoring sob contexto pending antes de confiar nos números `_refN` do eval.
