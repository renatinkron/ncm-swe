# M04 — Métricas honestas

**Objetivo**: saber quando uma métrica está medindo o que você quer —
e quando ela está medindo algo conveniente mas irrelevante.

## O imposto da métrica cômoda

Cross-entropy é fácil de computar e desce suavemente — por isso todo mundo
loga. Mas o objetivo real do projeto é **Python que compila**. Entre CE e
`ast.parse` há um abismo:

```
CE 5.17 → 3.2 (5 épocas)   ← sobe devagar, linear, bonito
exact-match fino held-out:  4.3% → 10.2%
parse rate dos samples:     ainda ~0%
```

CE caindo NÃO implica código compilável — cada ponto de CE compra menos
acurácia exata, e parse exige ~641/641 códigos certos (um erro de 1 token
já mata o parse: medido — 80% das lições falhavam com ~1.7% de corrupção).

## A hierarquia de métricas do projeto

| Métrica | Mede o quê | Decisão que habilita |
|---|---|---|
| CE de treino | fit estatístico | "está aprendendo algo?" — sanity |
| acc exact-match por nível (teacher-forced) | precisão por componente | "onde está a fraqueza?" |
| acc held-out vs treino | generalização vs memorização | "mais dados ou mais capacidade?" |
| recovery canal-only | teto do canal | "o canal é inocente?" |
| **ast.parse / compile** | **o objetivo real** | "funciona?" |

Regra: a métrica de decisão final é sempre a mais alta da hierarquia que
dá sinal. Enquanto parse = 0/16, as inferiores carregam a decisão — mas
ninguém comemora CE.

## Held-out vs treino — o par que nunca anda separado

Medição de 2026-11-16:

```
fine acc rows de TREINO:   18.3%
fine acc rows HELD-OUT:    10.2%
```

- Se treino ≈ held-out e baixo → **subajuste** (falta capacidade/épocas)
- Se treino >> held-out → **sobreajuste** (falta dados/regularização)
- Aqui: gap ~1.8x, ambos baixos → majoritariamente subajuste → continuar
  treinando ajuda, mas o muro real é capacidade+entropia

Sem o par, você não sabe se está olhando para um muro de capacidade ou
um muro de dados — e os consertos são opostos.

## Teacher-forced ≠ free-running

Teacher-forced (prefixo verdadeiro): fine acc 10.2% held-out.
Free-running (cada erro alimenta o próximo): **muito pior** — erros
compõem. Os samples são a evidência: esqueleto estrutural OK, tokens
corrompidos. Sempre reporte qual dos dois você mediu.

## Lab 4 — Auditar a métrica de uma conclusão

**Objetivo**: caçar métricas que não medem o objetivo.

```bash
cd /home/kron/Documents/projects/ncm_swe
# A métrica de treino:
tail -5 artifacts/ncm_code/train_code3.log
# O que ela NÃO mostra — as amostras reais:
python3 -c "import json; d=json.load(open('artifacts/ncm_code/live_samples.json')); print(d.get('epoch'), str(d)[:400])"
```

**Observe**: compare o CE da última época com o conteúdo textual dos
samples. Os dois contam a mesma história?

**Decida**: escreva a frase de status mais honesta possível com os dois
dados lado a lado. (Exemplo do próprio projeto: *"CE 3.2 e caindo; samples
com estrutura de programa mas superfície corrompida; parse ainda ~0%"*.)

---

*Anterior: [M03](m03-decomposicao.md) · Próximo: [M05 — Performance: profile antes de otimizar](m05-performance.md)*
