# M10 — Capstone: rumo ao código compilável

**Objetivo**: integrar os nove módulos numa visão do pipeline completo —
e nomear honestamente o que falta para o modelo chegar lá e para o
próximo salto de capacidade.

## O pipeline de ponta a ponta, com o estado real

```
corpus Python 168MB ─BPE(8k)─▶ tokens ─VQ encode─▶ codes_code.npy (20k×641)
                                                  │
  LevelGenerator ~120M ◀── warm-start gen_lessons2.pt
         │  treina: 8k rows, teacher-forced, batch 16, LR 1e-4
         ▼
  códigos previstos ─VQ decode (vq_code_tune, 96.8% rec)─▶ tokens ─BPE⁻¹─▶ texto
         │                                                              │
         └── verificador: ast.parse/compile ◀── decide ──────────────────┘
```

Estado medido em cada elo (2026-11-16):

| Elo | Estado | Evidência |
|---|---|---|
| Canal VQ | provado lossless | overfit 100%, recovery 96.8% |
| Canal-only E2E | sustenta 55.9% parse | 71/127 lições |
| Gerador | aprende, subajustado | fine held-out 10.2%, CE 3.2↓ |
| Decodificação | ponto de alavanca | best-of-N + repair = plano |
| Objetivo (parse livre) | ainda ~0% | samples: estrutura sem grafia |

## O que este livro cobriu — em uma frase cada

- **M00**: pesquisa com restrição exige método, não improviso
- **M01**: experimento que não decide nada é CPU queimada
- **M02**: sem save atômico + step log + amostras, o run não existiu
- **M03**: decomponha o pipeline — conserte o componente, não o sintoma
- **M04**: CE caindo ≠ objetivo avançando; métrica final é `ast.parse`
- **M05**: roofline primeiro — nosso treino já roda ~95-100% do silício
- **M06**: a máquina é do usuário; contenção se resolve com prioridade
- **M07**: negativo registrado = experimento que não se repete
- **M08**: o estado mora no disco; a sessão é só a janela
- **M09**: erro aleatório + verificador grátis = best-of-N + repair

## A próxima barreira — capacidade de fronteira

O muro nomeado pelas medições: **capacidade × entropia dos códigos finos**.
Opções ordenadas por alavanca medida:

1. **Extrair do que existe**: best-of-N + repair head (M09) — barato,
   usa o modelo atual, explora o erro aleatório
2. **Reformular o canal**: granularidade dos códigos finos — se a entropia
   por posição é o muro, mudar a representação pode comprar mais que
   qualquer quantidade de treino
3. **Escalar o gerador**: 120M → mais — caro em CPU, e a medição diz que
   a deficiência é difusa, não "quase lá"
4. **Mudar o regime de dados**: rejection-sampling FT sobre os acertos do
   próprio modelo — disponível quando parse >5-10%

A lição de método que fecha o livro: **a resposta certa muda quando a
medição muda.** O plano de hoje (decodificação) nasceu de um número que
não existia ontem (deficiência difusa, l0=0%). O plano de amanhã nascerá
do próximo.

## Lab 10 — Escrever a próxima entrada do handoff

**Objetivo**: praticar a decisão orientada a evidência de ponta a ponta.

```bash
cd /home/kron/Documents/projects/ncm_swe
tail -5 artifacts/ncm_code/train_code3.log          # CE atual
ls artifacts/ncm_code/                              # checkpoints vivos
sed -n '460,500p' SESSION_HANDOFF.md                # o que já se sabe
```

Redija a entrada de handoff que fecharia este caminho probatório: o que
foi provado, o que falta, e — usando a tabela de opções acima — qual é a
próxima ação e a métrica que julgaria seu sucesso.

**Decida**: se você só pudesse medir UMA coisa no próximo ciclo, qual
número diria mais sobre o caminho restante? Defenda com os dados.

---

*Anterior: [M09](m09-decodificacao.md) · [Índice](../README.md)*

---

## Apêndice — os comandos que acompanham o livro

```bash
# Monitorar o run vivo:
tail -f artifacts/ncm_code/train_code3.log

# Re-medir acurácia por nível em qualquer checkpoint/dataset:
podman run --rm --network none \
  -e THREADS=4 -e GEN_CK=/work/artifacts/ncm_code/gen_code2.pt \
  -e CODES=codes_code.npy -e OFFSET=8000 \
  -v "$PWD:/repo" -v "$PWD/artifacts:/work/artifacts" -w /repo \
  localhost/lcm-infer python3 scripts/_gen_acc.py

# O contrato de higiene de artefatos:
cat docs/ops/ARTIFACT_CLEANUP_RUN-040.md
```
