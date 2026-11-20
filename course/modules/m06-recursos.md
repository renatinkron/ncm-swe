# M06 — Governança de recursos

**Objetivo**: operar um treino longo numa máquina que também é o desktop
do usuário — sem derrubar a sessão gráfica, sem vazar para a rede, sem
encher o disco.

## Três eixos de governança

### 1. Isolamento — nada sai da caixa

Toda execução de treino/inferência no projeto roda em:

```bash
podman run --rm --network none \
  -v "$PWD:/repo" -v "$PWD/artifacts:/work/artifacts" \
  localhost/lcm-infer python3 scripts/ncm_codetrain.py
```

- `--network none`: nenhum pacote entra ou sai — reprodutibilidade e
  segurança (regra permanente do projeto para código de terceiros).
- `--rm`: o container morre com o processo; nada fica solto.
- Imagem local `lcm-infer`: versões de torch/MKL congeladas — o resultado
  de hoje é comparável com o de ontem.

### 2. Convivência — a máquina é do usuário primeiro

Diagnóstico real de 2026-11-15: desktop travando, load 15.9, idle 10%.
Causa NÃO era memória (18G livres) — era CPU: o trainer pedia 14 threads
num hardware de 8 núcleos/16 threads, deixando migalhas pro desktop.

A solução em camadas:

```
camada 1 — renice +15:      prioridade baixa → treino cede em qualquer
                            disputa com input/browser do usuário
camada 2 — affinity física: sched_setaffinity nos 8 núcleos FÍSICOS
                            (SMT não ajuda GEMM — FPU é compartilhada)
camada 3 — vizinhos:        o outro agente (vídeo) reniced +10, não morto
```

Resultado medido: mesmo ritmo de treino, idle saltou 10%→40%, desktop
voltou a responder. Lição: **contenção se resolve com prioridade e
topologia, não matando processos.**

Falha anotada: `--cpuset-cpus` do podman falha em rootless cgroup v2 —
o pinning foi feito por wrapper Python/`taskset` dentro do container.
`KMP_AFFINITY` emitiu warnings de CPU IDs inválidos no namespace do
container — registre o workaround, não só a intenção.

### 3. Higiene de disco — artefato ≠ registro

RUN-040 (esta mesma demanda): `artifacts/` chegou a 49G. Política
aplicada — registros, métricas, código e `final.pt` de cada linha ficam;
checkpoints intermediários e runs descartáveis saem:

```
49G → 17G   (32G liberados, manifesto em docs/ops/ARTIFACT_CLEANUP_RUN-040.md)
```

A regra duradoura: **um experimento é seus números, não seus terabytes.**
Se o `metrics.jsonl` e o `final.pt` sobrevivem, o experimento está
preservado — os `ckpt_step*.pt` são andaime.

## Anti-padrão que quase custou caro

Processo `sleep 21600` órfão de uma sessão antiga achado na auditoria de
lentidão — inofensivo, mas sintoma de higiene: **toda sessão longa deve
deixar rastro fechável** (shell id, pidfile, ou tmux nomeado).

## Lab 6 — Auditoria de convivência

**Objetivo**: encontrar quem disputa recurso com seu run.

```bash
# Quem come CPU, por prioridade:
ps -eo pid,ni,pcpu,comm --sort=-pcpu | head -15
# Load vs capacidade:
uptime; lscpu | grep -E "^CPU\(s\)|Thread|Core"
# Disco dos artefatos:
du -sh artifacts/*/ | sort -rh | head -10
```

**Observe**: load > núcleos físicos = fila; nice>0 nos jobs longos = o
desktop respira; artefatos grandes sem métrica ao lado = candidatos à
política de retenção.

**Decida**: qual único ajuste devolveria mais interatividade ao usuário
sem parar seu trabalho? No nosso caso foi renice+affinity — no seu pode
ser outro.

---

*Anterior: [M05](m05-performance.md) · Próximo: [M07 — Resultados negativos como ativos](m07-negativos.md)*
