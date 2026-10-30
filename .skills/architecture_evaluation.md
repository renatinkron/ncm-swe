# Architecture Evaluation Skill

**Goal**: avaliar uma mudança ou sistema como Especialista II em Engenharia e
Arquitetura, conectando algoritmos, memória, redes, falhas e limites de confiança
antes da decomposição das tarefas.

**Model**: `claude-sonnet-4-6`; escalar para `claude-opus-4-8` em avaliações de
alto risco, ambíguas ou críticas para segurança.

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Quando executar

Executar depois de `research` (quando aplicável) e antes de `spec_to_tasks` para
demandas standard/complex que envolvam arquitetura, dados, APIs, concorrência,
performance, confiabilidade, infraestrutura, autenticação, autorização ou
segurança. Não executar no fast path local e de baixo risco.

## Isolamento de contexto

Recebe somente `demand`, `scope`, `research_brief` opcional e o `codebase`/diagrama
relevante em modo somente leitura. Não recebe task plan, contrato, execução ou
conclusões de outro agente.

## Cinco símbolos de decisão

- **[■] ESTADO — onde o dado mora?** Mapear dono, ciclo de vida, durabilidade,
  consistência e recuperação; verificar perda após crash e necessidade de WAL,
  checkpoint, idempotência ou replay.
- **-> FLUXO — como o dado se move?** Mapear hops síncronos/assíncronos, filas,
  backpressure, congestionamento, ordem, retries, timeouts e duplicação.
- **⚡ COMPUTAÇÃO — onde processa?** Avaliar CPU, memória, disco, rede,
  alocações, working set, localidade, cache misses, serialização e context
  switches. Big-O não é promessa de latência; marcar medições necessárias.
- **✕ FALHA — o que quebra?** Enumerar perda de componente, falha parcial,
  timeout, overload, entrada corrompida e indisponibilidade; exigir fallback,
  circuit breaker, orçamento de retry, degradação ou fail-closed explícito.
- **🔒 TRUST BOUNDARY — onde a confiança muda?** Marcar fronteiras de processo,
  host, usuário, tenant, rede e credencial; separar identidade autenticada,
  autorização e integridade verificadas de meras suposições.

## Blocos de avaliação

Cobrir quando aplicável: realidade física de performance; fluxo distribuído e
composição de taxas/payloads/timeouts; confiabilidade e recuperação; segurança,
isolamento e egress; trade-offs, alternativas rejeitadas, riscos aceitos e o
menor teste que pode falsificar a decisão.

## Saída

Persistir `.agentic/research/{slug}-architecture-evaluation.md` com:

```markdown
# Architecture Evaluation — {slug}
## Context and assumptions
## System map
## [■] State
## -> Flow
## ⚡ Computation
## ✕ Failure
## 🔒 Trust boundaries
## Risks and blind spots
## Trade-offs and decision
## Validation plan
## Inputs for spec_to_tasks
```

Distinguir fatos observados, hipóteses e controles propostos. A seção final deve
fornecer restrições, critérios de aceitação e testes concretos, sem implementar
código nem aprovar silenciosamente uma hipótese não validada.

## Contract


```contract
inputs: demand, scope, research_brief (optional), codebase
outputs: .agentic/research/{slug}-architecture-evaluation.md (written_by: self)
requires_tools: Read, Write, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=arch_eval <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
