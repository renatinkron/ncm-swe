# RUN-010 Blind Review

Cold review: `scripts/train_token_head.py` (new), `spec_pyramid.py`
(--score tokhead), eval/bench JSONs, `artifacts/report.md` seção.

## Findings

- **info** — `train_token_head.py`: eval split usa o MESMO gerador
  `g` (seed 0) que o treino — evals consomem draws do stream de
  amostragem (determinístico, reproduzível, mas eval e treino não são
  streams independentes; aceitável para probe, documentar se escalar).
- **info** — `predicted_code_vecs` sob `no_grad` (correto — heads
  congeladas), mas chamado no loop de TREINO para a variante codes:
  códigos previstos no tempo de treino = casamento com inferência
  (decisão correta, explícita no docstring).
- **info** — `spec_pyramid.py`: `tokhead` é global setado em main —
  funciona, frágil a reuso por import (documentado pelo comment).
- **info** — `accs()` retorna contagens, divisão por batch no caller —
  ok.
- **verified ok** — `predicted_code_vecs` greedy sequencial usa as
  MESMAS camadas cond/out que a inferência (não há teacher forcing
  no input do classifier) — casamento treino/inferência real.
- **verified ok** — `--score tokhead` usa códigos do OFFSET-HEAD j
  (não do head 0) para o draft j — semântica correta (o codes head foi
  treinado em códigos de head de offset 0, mas a interface é a mesma;
  degradação com offset esperada e medida=0).
- **verified ok** — state-dict strip `codes.` → `net.*` carrega
  exatamente 4 tensores; eval() setado.
- **verified ok** — números do report.md batem com
  token_head_eval.json (feat 0/0, codes 3.1%/0) e hist no final.pt
  (acertos em @700/@1100/@1300).

## Patterns

n/a — achados info-level.

## Verdict

Sem blockers, sem majors. A ablação foi executada corretamente
(variantes isocapacitadas, mesmo forward, mesmo alvo) e a conclusão
(códigos carregam sinal que feat-only não extrai) é suportada por
perda E acurácia, não por anedota.
