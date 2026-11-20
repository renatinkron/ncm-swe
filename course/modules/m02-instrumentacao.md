# M02 — Instrumentação antes de otimização

**Objetivo**: internalizar que um run sem instrumentação é um run que não
aconteceu — e aprender o custo real de descobrir isso tarde demais.

## O incidente que ensinou tudo

Registrado como INCIDENTE 2 em `SESSION_HANDOFF.md` (2026-11-15):

Um run de 8k rows completou **duas épocas** (CE 4.347, prova em
`live_samples.json` timestamp 08:20) e foi morto num relaunch às 08:41.
O script antigo **só salvava no final** — as duas épocas evaporaram.
Pior: o novo run sobrescreveu o checkpoint, apagando até o estado inicial.

Horas de CPU transformadas em nada por causa de três linhas de código
que não existiam.

## O kit mínimo de sobrevivência

Depois do incidente, o protocolo do run passou a exigir:

```python
# 1. Save ATÔMICO por época — .tmp -> rename (nunca escreve direto)
torch.save(sd, tmp); os.replace(tmp, final)

# 2. Step log — STEP_LOG=50: progresso visível entre épocas
print(f"ep{ep} step {nb} loss={loss:.4f} t={elapsed}s", flush=True)

# 3. Amostras vivas — live_samples.json mostra O QUE o modelo escreve,
#    não só quanto vale a loss
```

Cada item responde a uma falha distinta:

| Instrumento | Falha que previne |
|---|---|
| save atômico/época | kill/OOM/reboot apaga horas de progresso |
| step log | "está aprendendo ou travou?" sem esperar 2h de época |
| live samples | CE caindo mas saída degradada — invisível na métrica |

## Por que ANTES e não durante

Instrumentação retroativa não existe: você não pode reconstruir as amostras
que o modelo produzia na época 2 depois que ela passou. A regra do projeto:

> Um run longo só é lançado quando o script (a) salva atomicamente por
> época, (b) loga por step, (c) deixa amostras inspecionáveis, e (d) tem
> snapshot do checkpoint anterior antes de sobrescrever.

Custo: ~20 linhas. Retorno: o run de 2026-11-16 sobreviveu a kill,
relaunch e três mudanças de configuração de afinidade sem perder nada.

## Bônus — idempotência importa

`train_code3.log` registra `t=` absoluto por linha; checkpoints levam
timestamp no nome (`gen_code2.ep3_20261115.pt`). Quando a sessão é
interrompida, qualquer pessoa — inclusive você daqui a uma semana —
reconstrói o que aconteceu lendo os artefatos, sem depender de memória.

## Lab 2 — Auditar seu próximo run

**Objetivo**: verificar instrumentação antes de lançar.

```bash
cd /home/kron/Documents/projects/ncm_swe
# Onde o script salva? É atômico? Por época ou só no fim?
grep -n "save\|replace\|tmp" scripts/ncm_codetrain.py
# O que o log mostra entre épocas?
grep -n "step\|print" scripts/ncm_codetrain.py | head -10
# Os artefatos têm timestamp que reconstrói a linha do tempo?
ls -la --time-style=+%H:%M artifacts/ncm_code/ | head
```

**Observe**: localize no script os pontos exatos dos três instrumentos do kit.

**Decida**: se você fosse matar o processo agora, o que se perderia? Se a
resposta for "mais que uma fração de época", a instrumentação falhou.

---

*Anterior: [M01](m01-ciclo-hipotese.md) · Próximo: [M03 — Diagnóstico por decomposição](m03-decomposicao.md)*
