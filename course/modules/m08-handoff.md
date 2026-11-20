# M08 — Handoff e continuidade

**Objetivo**: construir memória institucional que sobreviva a interrupções,
trocas de contexto e sessões de dias — porque pesquisa em CPU não cabe
numa tarde.

## O problema: sessões longas, contexto finito

Um run de 15 épocas × ~2h = ~30h de treino espalhado por dias. Entre
perguntas do operador, travamentos de desktop, relaunches e noites de
espera, o "estado mental" do experimento é o primeiro a se perder.
O handoff resolve: **o estado do experimento mora no disco, não na cabeça.**

## O arquivo que é parte da definição de pronto

`SESSION_HANDOFF.md` neste projeto é obrigatório (AGENTS.md): uma tarefa
com mudança material **não está concluída se o handoff está desatualizado**.

Anatomia de uma boa entrada (do arquivo real):

```markdown
## MEDIÇÃO 2026-11-16 — acurácia por nível no corpus (decisiva)
Setup: _gen_acc.py (GEN_CK, CODES, OFFSET). Teacher-forced, 64 held-out.
- fine: 4.29% -> 10.16% | l1: 44.9% | l0: 0%
- treino vs heldout: 18.3% vs 10.2% → subajuste
Decisão: deficiência difusa → decodificação > loss ponderado
```

Cada entrada responde: **o que medimos, com o quê, o que deu, o que
decidimos** — em ordem cronológica, sem reescrever o passado.

## Regras que mantêm o handoff honesto

1. **Append, não reescrita** — quando uma medição é superada, marca-se
   "superado por X", não apaga-se. A trajetória de crença é dado.
2. **Negativos entram com números** — ver M07.
3. **Instruções permanentes também** — o usuário pediu monitoramento
   contínuo; está escrito no handoff como INSTRUÇÃO PERMANENTE com data.
   Preferências operacionais são estado do projeto, não da conversa.
4. **Incidentes entram com lição** — o checkpoint perdido virou INCIDENTE 2
   + a lição que gerou o protocolo de save atômico (ver M02).

## O monitoramento como estado, não como atenção

"Manter a sessão ativa" não significa olhar a tela — significa um loop
que verifica o log, registra marcos e volta a dormir. O padrão usado:

```
ciclo: tail do log → comparar com último marco → se novo step/época,
anotar → sleep ~5min → repetir
```

O monitor nunca bloqueia respostas ao usuário — a instrução permanente
deixa explícito: **o acompanhamento não para por causa de conversa
paralela**. Ambos acontecem porque o estado está no log, não na sessão.

## Lab 8 — Simular uma entrega de bastão

**Objetivo**: testar se o handoff sustenta uma retomada fria.

```bash
cd /home/kron/Documents/projects/ncm_swe
# Finja que você chega AGORA sem nenhum contexto. Em 5 minutos, usando
# só o disco, responda:
sed -n '424,500p' SESSION_HANDOFF.md   # a seção do run atual
tail -3 artifacts/ncm_code/train_code3.log
```

Perguntas para responder: o que está rodando? Em que época? O que já foi
provado? O que foi refutado? Qual a próxima ação planejada?

**Observe**: se alguma resposta exigisse memória de conversa, o handoff
falhou naquele ponto.

**Decida**: escreva a entrada de handoff que VOCÊ deixaria hoje se a
sessão pudesse ser cortada em 10 minutos.

---

*Anterior: [M07](m07-negativos.md) · Próximo: [M09 — Decodificação como multiplicador](m09-decodificacao.md)*
