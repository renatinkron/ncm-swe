# Blind Review — RUN-044

Escopo: `scripts/_func_units.py`, `scripts/_eval_func.py`, `scripts/_gold_func_channel.py`, diffs em `scripts/ncm_codetrain.py`.

## Findings

### minor — `_func_units.py`: blocos col-0 só capturam def/class top-level
`START` casa só coluna 0 — funções aninhadas e métodos viram parte do bloco pai (correto para standalone), mas `if __name__ ==` e código top-level solto ficam fora (ok, é o recorte desejado). Limitação real: decoradores (`@app.route`) na linha ANTES do def não são capturados — unidade `def` decorada vira `def` sem decorator: ainda parseia (sem NameError em ast.parse pois decorator é só nome), mas muda semântica. **Impacto**: dataset levemente diferente do código real; para o probe de parseabilidade não bloqueia. Recomendo nota.

### minor — `_func_units.py`: fim de bloco inclui comentários top-level do próximo bloco
Linhas `#` na coluna 0 entre funções são absorvidas na unidade anterior (condição `ln[0] not in " \t#"`). Inofensivo para parse, mas unidades carregam comentários do próximo bloco. **Impacto**: ruído leve nos dados.

### minor — `_eval_func.py`: `filler_fraction` retorna 0.0 em tokenize failure
Saídas corrompidas (a maioria) falham no tokenize → filler 0.0, o que *subestima* filler em saídas degeneradas. Infla aparente "limpeza" do argmax (filler 0.0 porque texto não tokeniza, não porque não há filler). **Impacto**: métrica otimista no pior caso; documentar.

### minor — `_eval_func.py`: "argmax" é top_k=1 via multinomial
`gen.generate(temperature=1.0, top_k=1)` amostra de 1 candidato = determinístico, ok — mas o rótulo "argmax" no JSON pode induzir leitor a achar que é argmax puro sobre logits (é equivalente, correto). Cosmético.

### minor — `ncm_codetrain.py` MAX_T: corte só entre batches
`if max_t and time.time()-t0 > max_t` checado por step — overshoot de até ~60s/step, aceitável. Também salva checkpoint ao sair do inner loop (bom). Porém `res["ce_last"]` fica com a média da época truncada rotulada como `ep0 loss=` — leitor pode confundir 3.07 com "convergiu em 3.07". Documentado no report.

### minor — `_gold_func_channel.py`: `_ok` definida depois de main() mas chamada dentro
Funciona (def executa antes da chamada em `__main__`), mas ordem confusa. Cosmético.

### observação — alinhamento codes↔units depende de N_ROWS
`codes_func.npy` tem prefixo 0:10000 de units; qualquer eval que pegue `units[-n:]` desalinha (já pegou o autor uma vez). Footgun real: quem reutilizar codes_func.npy deve lembrar que é PREFIXO. **Sugestão**: metadado `codes_func.meta.json` com `{n_units_encoded, units_file}` — não implementado, registrado.

### observação — gen_func1.pt (479MB) é produto de 232 steps
Checkpoint de meia-época; rotulá-lo "gen_func1" sem sufixo de passos pode induzir reuso como "modelo func-units" final. Documentado no handoff.

## Resumo

Nenhum major/blocker. Footguns de alinhamento N_ROWS e de filler_frac em texto não-tokenizável registrados. Métricas centrais (0/32, canal 8/32, text_sim 0.92) verificadas contra os JSONs persistidos.
