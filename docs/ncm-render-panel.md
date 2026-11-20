# NCM render-panel — design e medidas (RUN-048)

Painel de "renderização" sobre o checkpoint oficial do NCM (TinyStories,
88.6M params gerador + 36.4M VQ-VAE), rodando em CPU comum via container
`lcm-infer` (`--network none`). Implementação: `scripts/ncm_panel.py`.

## Mapeamento renderizador → NCM

| Renderizador | NCM | Knob |
|---|---|---|
| Passadas de refinamento | 12 níveis fixos `[1,8,16,24,32,40,52,64,76,92,108,128]` | `--preview` mostra cada nível; early-exit = trabalho futuro |
| Amostras (samples) | `--n` gerações independentes | `--n`, `--seed` |
| Sampler | temperatura/top-k/top-p por nível | `--temp --top_k --top_p` |
| Guidance | CFG por nível (escala cresce com o nível) | `--cfg` |
| Nível 0 (conceito) | fonte do código bruto: marginal do corpus, uniforme, suavizada | `--l0 unigram|uniform|active|smooth:ε` |
| Filtros (box/Lanczos…) | penalidades anti-repetição no decode final | `--freq --mu --adj a,w,d` |
| Preview vs final | decode por nível vs decode final | `--preview` |

"Bounces" (re-amostragem recursiva) e early-exit por nível **não** existem no
ckpt oficial (`predict_level0=false`, `query_level0=false`): documentados como
trabalho futuro, não simulados.

## Medido nesta máquina (Ryzen 7 4750U, ~37 GB/s, 8 threads)

| config | s/sample | notas |
|---|---|---|
| defaults (`unigram`, t=1.0, k=50, p=0.9) | **0.527** | 12 passes, texto TinyStories coerente |
| `smooth:0.05`, t=0.8, k=40, p=0.95, **cfg=1.5**, freq=0.5, adj | 0.958 | CFG dobra o batch → ~2× custo; métricas estáveis |

Métricas do smoke (n=4): dup4_rate 0.0, unicode_frac 0.002, distinct_ratio 0.65,
mean 530 chars. O checkpoint oficial já produz texto legível — bem melhor que o
`bench_official.json` antigo (gerador treinado por nós, não o oficial).

## Módulos LoRA (plano)

- Alvo: projeções `c_attn`/`c_proj` dos blocos do `NextConceptTransformer` +
  opcionalmente `lm_head`/cabeças de nível. Módulos = adapters treinados sobre
  `data/lora_dump/knowledge_dump.jsonl` (91 entradas: programação EN;
  lógica/texto EN+PT-BR).
- Implementação nesta passada: **hand-rolled** (peft ausente no venv e na
  imagem). ~60 linhas: congelar base, treinar apenas ΔW = BA de baixo rank.
  Upgrade path: imagem com `peft` quando módulos se multiplicarem.
- Domínios pretendidos: `python`, `java`, `math`, `web-research` (ferramenta,
  não peso), `logic-pt`, `writing-pt`. Um módulo por família de dados.

## Limites honestos

- O ckpt é **TinyStories** (~85M): texto infantil coerente, não código nem
  conhecimento avançado. O painel mede *controlabilidade*, não utilidade.
- Qualidade ≠ progresso: dup4_rate 0.0 não significa bom texto; requer eval
  semântico separado (próximo passo).
- CFG aqui é o do upstream (escala por nível sobre batch [doc;null]); o null é
  prefixo EOS+pad — funciona, mas é "classifier-free" fraco vs. null treinado.
- CPU: ~0.5–1 s/amostra; viável para REPL, não para serving.

## Reprodução

```bash
podman run --rm --network none -v $PWD:/repo -v $PWD/artifacts:/work/artifacts \
  -w /repo -e THREADS=8 -e NCM_CKPT=/work/artifacts/ncm_official \
  localhost/lcm-infer:latest python3 scripts/ncm_panel.py \
  --n 4 --preview --out /work/artifacts/panel_smoke.json
```

Testes: `~/.venvs/ncm/bin/python -m pytest tests/test_panel.py tests/test_dump.py`
(21 testes, helpers puros + schema do dump; ckpt não necessário).
