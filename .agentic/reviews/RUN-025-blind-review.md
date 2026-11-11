# Blind Review — RUN-025 (revisor frio, agente e52360c9)

## Veredicto: 0 blocker, 0 major — 6 minor + 4 nit

## Disposição

| finding | severidade | disposição |
|---|---|---|
| corrupt_schedule/hi/lo só via getattr — rebuild de ckpt cai pra 'flat' | minor | **FIXED** — agora args do ctor com defaults que preservam semântica anterior |
| forward aninhado roda em train() → predições com dropout (dormente: dropout=0) | minor | **FIXED** — self.eval() + restore em finally |
| outs fp32 (B×ΣL×V) + softmax temp vivo | minor | **FIXED** — del outs após amostragem |
| eval_ckpt_v4: sleep único, mtime não re-checado → race residual | minor | **FIXED** — loop re-stat (v3 e v4) |
| MAXSEC 60h < runtime projetado v4 | nit | **FIXED** — 72h |
| contagem de "pontos flat" inconsistente no cycles file (1.048 in/out) | nit | **FIXED** — convenção "flat = 0.95–1.20 inclusivo" anotada |
| flat branch aloca rand(idx.shape) por nível (pré-existente) | nit | DEFER — mudar alteraria o RNG stream (compat) |
| self.forward bypassa hooks __call__ | nit | FIXED incidental — trocado p/ self(idx) |
| label `[0-9]*` aceita sufixo; label explícito pode mislabel ckpt velho | nit | DEFER — misuse manual apenas, documentado |
| corrupt-hi/lo gravados no ckpt config p/ flat (dead values) | nit | DEFER — cosmético |

Verificados limpos pelo revisor: recursão gated por corrupt=0.0 +
finally; no_grad herdado do forward externo; alinhamento preds[j-1]↔
outs[i]↔chunk_levels[i]↔nível j exato; OFFSET/FIRST aritmética; todos
os números tabela↔seção do cycles file re-computados corretos.
