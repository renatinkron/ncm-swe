# M00 — O caminho probatório

**Objetivo**: entender por que este livro existe, de onde vem o material e
como transformar um repositório de pesquisa vivo em didática.

## O problema que este livro endereça

A maior parte do material sobre ML assume datacenter: GPU sobrando, cluster
de treino, inferência via API. Este projeto corre na direção oposta —
responder, com rigor, à pergunta:

> *Até onde vai um modelo treinado inteiro num desktop comum, e o que o
> processo de chegar lá ensina sobre fazer ciência com restrição?*

A restrição não é um detalhe — é o que torna o método necessário. Quando cada
época custa horas e cada GB de disco disputa com o sistema do usuário, você
não pode se dar ao luxo de experimento mal desenhado, métrica ambígua ou
checkpoint perdido. O método nasce da escassez.

## O material: um repositório, não uma narrativa

Tudo que este livro ensina aconteceu de verdade neste repo
(`ncm_swe`), entre outubro e novembro de 2026:

- Um pipeline NCM (VQ-VAE + LevelGenerator) treinado para emitir código
  Python compilável a partir de códigos discretos.
- Medições públicas: `SESSION_HANDOFF.md` acumula cada resultado — inclusive
  os negativos — com números, não adjetivos.
- Falhas documentadas: checkpoint de 2 épocas perdido por falta de save
  incremental, hipótese de distilação refutada por baseline aleatória,
  otimização de compilador que não moveu nada (e a matemática que explicou
  por quê).

Cada capítulo cita o artefato real onde o episódio está registrado. O livro
não reconta a história — ele a estrutura em lições.

## O contrato com o leitor

1. **Todo número é rastreável.** Se um capítulo diz "65% → 96.8%", existe
   uma entrada de handoff ou um `metrics.json` onde isso está escrito.
2. **Nenhum resultado negativo é escondido.** Hipóteses refutadas são
   material didático de primeira linha, não vergonha.
3. **Nada é prometido além do medido.** Onde o modelo ainda não gera Python
   compilável, o livro diz exatamente isso — e mostra o que falta.

## Lab 0 — Ler o terreno

**Objetivo**: reconhecer a anatomia de um repo de pesquisa honesto.

```bash
cd /home/kron/Documents/projects/ncm_swe
# A memória acumulada do projeto:
grep -n "^## " SESSION_HANDOFF.md
# Os artefatos vivos do experimento atual:
ls -la artifacts/ncm_code/
# O log do treino em andamento:
tail artifacts/ncm_code/train_code3.log
```

**Observe**: o handoff é organizado por RUN/sessão; resultados negativos têm
seção própria com a mesma dignidade dos positivos; o log do treino registra
step e tempo, não só loss.

**Decida**: se você herdasse este repo hoje, conseguiria reconstruir o estado
mental do pesquisador anterior? O que faltaria?

---

*Próximo: [M01 — O ciclo hipótese→evidência](m01-ciclo-hipotese.md)*
