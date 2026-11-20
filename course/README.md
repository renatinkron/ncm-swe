# O Ofício da Pesquisa Empírica em IA Local

> Livro interativo em construção — será lançado ao final do *caminho
> probatório* do projeto ncm_swe. Cada capítulo usa **casos reais medidos
> neste repositório** como material didático, incluindo resultados negativos.

## O que este livro ensina

Como conduzir pesquisa aplicada de ML em hardware comum — um desktop com
CPU consumer, sem GPU, sem cluster — mantendo rigor experimental:
hipóteses testáveis, baselines honestas, instrumentação antes de otimização,
resultados negativos como evidência, e continuidade entre sessões.

O projeto de referência: treinar um gerador de código NCM (VQ codes →
Python compilável) em um Ryzen 7 PRO 4750U com 32GB RAM.

## Como ler

- Capítulos curtos; cada um fecha com um **Lab** — exercício executável com
  artefatos reais deste repositório.
- Números citados são medições registradas em `SESSION_HANDOFF.md` e em
  `.agentic/state/runs/` — não são ilustrativos, são reproduzíveis.
- Resultados negativos aparecem com o mesmo peso dos positivos: eles são
  o método funcionando.

## Módulos

| # | Módulo | Pergunta que responde |
|---|---|---|
| 00 | [O caminho probatório](modules/m00-caminho-probatorio.md) | Por que este livro existe e como usar casos reais |
| 01 | [O ciclo hipótese→evidência](modules/m01-ciclo-hipotese.md) | Como estruturar um experimento que decide algo |
| 02 | [Instrumentação antes de otimização](modules/m02-instrumentacao.md) | Por que log, checkpoint atômico e amostra viva vêm primeiro |
| 03 | [Diagnóstico por decomposição](modules/m03-decomposicao.md) | Como isolar canal, gerador e decoder para achar o gargalo real |
| 04 | [Métricas honestas](modules/m04-metricas.md) | Held-out vs treino, teacher-forced vs livre, CE vs parse |
| 05 | [Performance: profile antes de otimizar](modules/m05-performance.md) | Roofline, py-spy, e por que torch.compile deu ~0% |
| 06 | [Governança de recursos](modules/m06-recursos.md) | Containers isolados, afinidade, prioridade, higiene de disco |
| 07 | [Resultados negativos como ativos](modules/m07-negativos.md) | kdistill refutado, probe l0 negativo — o que eles ensinaram |
| 08 | [Handoff e continuidade](modules/m08-handoff.md) | Memória institucional: SESSION_HANDOFF e sessões longas |
| 09 | [Decodificação como multiplicador](modules/m09-decodificacao.md) | Best-of-N + verificadores determinísticos + repair |
| 10 | [Capstone: rumo ao código compilável](modules/m10-capstone.md) | O pipeline completo e a próxima barreira (capacidade) |

## Convenções dos Labs

Cada lab indica: **objetivo**, **comando/artefato**, **o que observar** e
**o que decidiria** com aquele dado. Rode os labs de leitura direto no host;
labs que executam o modelo usam o container `localhost/lcm-infer`
(`podman run --rm --network none`), mesma convenção dos scripts em `scripts/`.
