# Blind Review — RUN-045

Escopo: `scripts/_bpe_probe.py` (diffs), `scripts/_eval_bpe_prefix.py`, artefatos de métricas.

## Findings

### minor — `id_to_token` pode retornar None
`fast.id_to_token(i) or ""` trata None como string vazia — silencia ids fora do vocab em vez de sinalizar. Para eval é aceitável (skip silencioso), mas um token desconhecido vira char ausente e pode mudar parse. **Impacto**: viés otimista leve; registrado.

### minor — `_BMAP` cobre só ranges 0-32 e 127-160
Byte-level GPT-2 mapeia exatamente esses (0-32, 127-160 → +256); o resto (33-126, 161-255) já é identidade — correto. Mas se o tokenizer usou normalização diferente (ex.: byte_fallback), caracteres fora do mapa passariam crus. Round-trip gold validou o caminho real, então ok para este tokenizer.

### minor — eval de geração não fixa `model.eval()` no caminho EVAL_ONLY
`generate()` chama `self.eval()` — ok. Mas o loop held-out usa `model.eval()` explicitamente antes; consistente. Sem dropout no modelo, irrelevante. Cosmético.

### minor — geração seeded sempre `"def "` — não testa `class `
O dataset contém units começando em `class`/`async def`; eval só sementeia `def `. Medição cobre o caso majoritário (~85% def) mas `class `-seed ficou não medido. **Sugestão**: regime `class_prefix` no próximo eval — não bloqueia.

### minor — held-out CE/top-1 contam PAD/EOS
Targets incluem posições de PAD pós-EOS — o modelo aprende a prever PAD (fácil), inflando top-1 levemente. Comparação RUN-043 vs RUN-045 sofre o mesmo viés em ambos → delta 39→77% segue válido direcionalmente. **Impacto**: número absoluto otimista; nota registrada.

### observação — `bpe_units_rows.npy` é cache silencioso
Se `func_units.txt` mudar, o cache velho seria reutilizado (stale). Mesmo footgun do `bpe_rows.npy` original. Registrado.

### observação — argmax colapso e truncamento são limites do modelo/oráculo, não bugs
Documentados no veredito; ctx-128 é hard cap do BPELM.pos.

## Resumo

Nenhum major/blocker. O fix de decode foi verificado contra round-trip gold e contra re-medição do modelo antigo (0/32 mantido → bug não explicava a falha anterior). Métricas centrais reproduzíveis via JSONs persistidos.
