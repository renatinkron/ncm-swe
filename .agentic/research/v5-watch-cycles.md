
## v5-ext ciclo 1 — @12499 (+500 steps, 12 nov)

| métrica | v5 @11999 | v5ext @12499 |
|---|---|---|
| dup_rate | 0,053 | **0,036** (−32%, >4σ — primeiro a romper o platô ~0,05) |
| distinct_ratio | 0,67 | **0,753** |
| exposure gap | +0,909 | +0,815 (Δ−0,09 inconclusivo em n=8) |
| gt CE | 4,217 | 4,235 |
| passes | 24 | 24 |

Leitura: dup_rate saiu do platô invariante de todos os braços —
treino intra+mix continuado move o stutter residual (não é sampling).
Gap estável/tendendo. Próximo @12999 ~2,2h.

## v5-ext ciclo 2 — @12999 (+1000 steps, 12 nov)

| ckpt | dup | gap | gt |
|---|---|---|---|
| v5 @11999 | 0,053 | +0,909 | 4,217 |
| @12499 | 0,036 | +0,815 | 4,235 |
| @12999 | 0,0555 | +0,751 | 4,320 |

**Correção do ciclo 1**: dup 0,036→0,0555 — o "plateau break" era
parcialmente sorte de seed (n=8 oscila ±0,02). Rebaixo para
"indício não confirmado — aguarda n=32".

Leitura: gap 0,91→0,82→0,75 descendente consistente, MAS por razão
mista — self CE melhora (5,13→5,07) enquanto gt CE degrada
(4,22→4,32): assinatura do trade-off corrupção↔limpeza (WPS).
Parte do ganho de gap é contábil. Decisão: manter treino até @14500
— se gt seguir subindo, o "ganho" é majoritariamente artefato;
se estabilizar com gap <0,7, é ganho real. Próximo @13499 ~2,2h.
