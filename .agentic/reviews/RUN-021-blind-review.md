# Blind Review — RUN-021

**Object under review:** `/home/kron/Documents/projects/ncm-local-study` — self-contained
academic publication repo (PT-BR + EN-US papers, TIMELINE, SANITIZATION log, README,
LICENSE, CITATION.cff, `.gitignore`, `artifacts/metrics/*.json|jsonl`,
`artifacts/samples/`, `reviews/RUN-021-*.md`, `paper/SOURCES.md`).
**Method:** cold read of every file; every headline number spot-checked against the
artifact mapped in `SOURCES.md`; repo-wide sweep for personal identifiers/secrets;
PT↔EN cross-read; referenced-file existence check.

Context note: the four `reviews/RUN-021-*.md` files quote a *superseded draft*
(they cite `held_l11 = 2,68`, "quadruplicar", "três máquinas", "±20 %" — none of
which survive in the current papers). Most of their majors were addressed in the
current text (contamination window extended to Phase A, Phase-D slower-than-AR
result now reported, `metrics_gen_ncm_*_final.jsonl` added, repetition-diagnosis
hedged). Findings below are only what remains wrong *now*.

## Findings

### Blockers

1. **EN abstract asserts the headline result in the wrong direction.**
   `paper/paper_en-us.md:11` — "the finest level converged **below** the reference
   model's own loss". Artifact says held_l11 = **2.7924**
   (`artifacts/metrics/metrics_gen_ncm_v2_final.jsonl:1`) against a reference of
   ~2.75 → 2.79 is *above*, not below. Everywhere else the claim is correct and
   hedged ("essentially at ... (~2.75)": `paper_pt-br.md:11`, `paper_pt-br.md:107`,
   `paper_en-us.md:104`, `SOURCES.md:26`, `README.md:31`). This is a leftover from
   the pre-revision draft (which reported 2.68 < 2.75 — see `reviews/RUN-021-review-1.md:30`,
   `RUN-021-review-2.md:155`). It falsifies the abstract of the EN paper, contradicts
   the PT version *and* the EN paper's own §4.3, and overclaims the study's central
   positive result. Must fix before publishing.

### Majors

2. **EN undercounts contaminated rounds — contradicts itself and PT/README.**
   `paper/paper_en-us.md:11` ("invalidated **four** full rounds of verdicts") and
   `paper_en-us.md:29` ("**four** entire rounds ... contaminated") vs
   `paper_en-us.md:73` itself ("every round that loaded a pretrained VQ-VAE —
   phases A (001–004), C (014–016) and D (017)" = **8 rounds**),
   `paper_pt-br.md:74` (same enumeration; "todos os vereditos de qualidade ... nessas
   rodadas estavam comprometidos") and `README.md:27-29` ("(001–004, 014–017);
   vereditos de qualidade comprometidos"). Even under the charitable reading
   ("only 014–017 produced quality verdicts"), the EN file contradicts its own
   §3.3 enumeration. Same vintage as blocker 1 — stale draft phrasing.

3. **SOURCES.md maps several cited numbers to artifacts that do not contain them.**
   The repo's selling point is number→artifact traceability; these mappings fail:
   - "VQ-VAE held_ce 0,152" → `eval_ts.json` has no `held_ce` field at all (only
     py_parse_rate / mean_words / distinct_word_ratio). `SOURCES.md:19`.
   - "rank mediano ~43.505/151.936" → `codepath_eval.json` stores per-position
     token IDs; there is no rank field. V=151936 confirmed (`pyramid_cb_qwen05_report.json:47`)
     but the median is unverifiable. `SOURCES.md:16`; claim in papers at
     `paper_pt-br.md:133` / `paper_en-us.md:130`.
   - "platô ~6,1" → `bench_ts_resume.json` has no loss field. `SOURCES.md:22`;
     claim at `paper_pt-br.md:99` / `paper_en-us.md:96`.
   - Taper "held CE 0,654, utilização 100%" — cited in `paper_pt-br.md:99` /
     `paper_en-us.md:96`; absent from `SOURCES.md` and from every artifact.
   - Mimir "0,07 conceitos/s" (`paper_pt-br.md:45`, `paper_en-us.md:43`,
     `TIMELINE.md:13`) — RUN-008 has no artifact at all.
   - l0 trajectory "8,36→7,97 in training windows" (`paper_pt-br.md:137`,
     `paper_en-us.md:134`) — `metrics_gen_ncm_v2_final.jsonl` holds only the
     step-3000 final record; no trajectory is published.
   - "×1,05–1,34" range (`paper_pt-br.md:86`, `README.md:23`, `TIMELINE.md:10`) —
     `bench.json:57` records only `speedup_wall: 1.14`; the endpoints come from
     RUN-001/003 benches that are not shipped.

4. **held_ce direction contradiction still unresolved.**
   `paper_pt-br.md:99` / `paper_en-us.md:96`: tapering "improved reconstruction
   (held CE 0.654, 100% utilization)". But `SOURCES.md:19` reports the *base*
   VQ-VAE at held_ce 0.152. 0.654 > 0.152 — worse, if the metric is the same.
   Either the metric/split changed between runs or "improved" refers to
   utilization only; the text does not say which. Flagged by `RUN-021-review-1.md:40`,
   still present.

5. **Sanitization log falsified: local paths survive in 5 artifacts.**
   `SANITIZATION.md:9` claims 0 occurrences of personal paths, but
   `"model_dir": "~/models/qwen2.5-0.5b-sft"` appears in
   `codepath_eval.json:3`, `pyramid_bench.json:3`,
   `spec_pyramid_path_bench.json:3`, `spec_topk_ar_bench.json:3`,
   `spec_topk64_ar_bench.json:3`. `~/` is a home-relative local path — no username
   leaks, but it is precisely the class of leftover the sweep claimed to clear.
   Same files also reference `artifacts/*.pt` paths whose files are `.gitignore`d
   (consistent with "checkpoints not distributed", but the dangling paths should
   be noted). Either scrub the configs or correct the sanitization log.

### Minors

6. **Parameter-count labels don't match artifacts.** `bench.json:27` records NCM
   params 68,834,656 and AR 67,151,232 — but `paper_pt-br.md:62`/`en-us:61` say
   "~62M", `TIMELINE.md:10` says "NCM 62M vs AR 67M", table row says "~62–68M"
   (doesn't cover 68.8M), and `SOURCES.md:11` says "AR 60M". Taper/intra arms are
   67,624,160/67,625,312 (`bench_ts2.json:27`, `bench_ts_intra.json:27`) labeled
   "~69M". Pick labels that bracket the measured values.
7. **"90% of neighbor positions" overstates the artifact.** `boundary_ce_intra.json`
   max boundaries are b28/32 = 87.5 %, b57/64 = 89.1 %, b115/128 = 89.8 %
   (`paper_pt-br.md:101`, `paper_en-us.md:98`). "~88–90 %" is the honest label.
8. **k5 mis-rounded.** `level_ablation.json:33` k5 `token_accuracy` = 0.3628 =
   36.28 % → `SOURCES.md:20` reports "36,4 %"; should be 36,3 %. (k6 0.9697 →
   "96,9 %" is truncation, acceptable.)
9. **"~×2,9 per token-equivalent" under-computes.** From `bench.json`: mean
   tokens_equiv 341.9 → (341.9/127)×(0.609/0.535) ≈ 3.07. `paper_pt-br.md:86` /
   `en-us:85`. The "~" hedges it; could still read ~×3.
10. **`SOURCES.md:17` conflates two different metrics** — "codes top-1 truth
    3,1%→2,34%" mixes `a1 = 3.125 %` (`token_head_ar_eval.json:9`) with
    `t1 = 2.34 %` (`token_head_eval2.json:7`) as if one progression. Also the
    "uniforme (11,9)" figure is derivable (ln 151936) but appears in no artifact.
11. **Author-initial inconsistency.** `SOURCES.md:48` lists "LI, **Z**. et al."
    for MC-SMoE; the papers' own reference lists correctly say "LI, **Pingzhi**
    et al." (`paper_pt-br.md:181`, `paper_en-us.md:178`).
12. **Run-018 double identity.** `TIMELINE.md:18` folds RUN-018 into "retreino
    limpo (v2)"; the numbering note (`paper_pt-br.md:70`, `en-us:69`) says 018 was
    an infrastructure round with no experiment; §3.3 says the fix landed "na run
    018" and "019/020 is the first clean test". Reconcile (e.g., "018 = fix/infra,
    019–020 = clean retrain").
13. **Malformed artifact.** `artifacts/samples/representative_samples.json:28` —
    `ts_word_salad.sample_0` is a truncated Python-`repr` dump of `bench_ts.json`'s
    config dict ("{'config': {... 'guard r"), not a generated-text sample.
    Embarrassing sloppy file in a publication repo.
14. **"same machine" claim unverifiable.** `paper_pt-br.md:111`/`en-us:108`
    compare v2's 1.306 s to "AR 0.617 s on the same machine" — but
    `bench_official.json`/`bench_v2.json` carry no `hardware`/`cpu_count` field
    (0.617 comes from `bench_ts.json:9`, cpu_count 16). Add the field or soften.
15. **Mixed-corpus AR baseline degeneration undeclared.** `eval.json` and
    `eval_471m.json` show AR `py_parse_rate` 0.0, `distinct_word_ratio` 0.25–0.29,
    and the AR samples in `bench.json`/`bench_471m.json` are word salad too —
    rows 1–2 of Table 1 compare latencies between *two* degenerate models. The
    paper is honest about this asymmetry for TinyStories but never says it for
    the mixed corpus.
16. **Private-repo internals leaked in a public review file.**
    `reviews/RUN-021-review-2.md:60-71` names the private sister repo `ncm_swe`,
    quotes its internal file paths (`src/ncm_swe/model/generator.py`,
    `scripts/train_generator.py`), commit SHAs (`000a476`, `e5cdec7`), and states
    the init-bug fix "exists only uncommitted in the working tree". Not a secret,
    but it discloses a second repo's structure and git hygiene in a repo meant to
    stand alone. Decide deliberately whether to keep.
17. **EN file typos.** `paper_en-us.md:63` stray `), — ` punctuation;
    `paper_en-us.md:15-17` "## Resumo (Portuguese abstract)" is a pointer to the
    PT file rather than a translated resumo (acceptable, but conspicuous in the
    "full English version" the README advertises).
18. **Provenance of "~2.75 reference" still thin.** Hedged correctly as
    "approximate-protocol" (`paper_en-us.md:104`, limitation vii), but never says
    whether it's the original paper's number or a local eval of the official
    checkpoint — one clause would close it.

### Info / trajectory

- `RUN-021.events.jsonl`: loop is clean — begin → issue_create → research →
  arch_eval → task_plan → execute → test(nc:0) → history. Two `step_failed`
  notes (task_plan gate content, missing execution_report) were logged honestly
  and retried; no self-reported passes without artifacts, no missing dispatches.
- Verified OK: all speedup figures (1.14, 2.27, 2.82, 3.22, 3.21, 1.22) match
  `speedup_wall` fields exactly; per-sample latencies, level passes (6→10),
  127 AR steps, params 478.6M/477.6M, RSS 5.7 GB, 14.5 h (52 103 s) all match;
  held metrics match `metrics_gen_ncm_*_final.jsonl` exactly (7.5453→"7,55",
  6.8095→"6,81", 8.2037→"8,20", 5.8158→"5,82", 2.7924→"2,79", 8.2174→"8,22" —
  and `held_loss` equals the mean of the 12 per-level held values in both files);
  boundary CEs match to 2 decimals; 0/40 codepath hits, accept 0@path /
  0.58@k=64, ~2×-slower spec arm (3.71 vs 1.82 tok/s), top-1 32.1 %, ~34 %
  pyramid slowdown, uniform CE 8.32 = ln(4096) — all confirmed.
- No `/home/`, `/Users/`, `/root/`, usernames, hostnames, or secrets anywhere
  outside `.git`; `renatinkron` in the README BibTeX is the declared public repo
  URL (intentional per `SANITIZATION.md:20-21`). LICENSE is a genuine CC BY-NC
  4.0 text; CITATION.cff consistent (2026-11-09).
- `reviews/` files cite a superseded draft — fine as process history; their
  line references no longer resolve to the current text.

## Patterns

- **Unexported eval metrics:** 6 of the 7 traceability gaps in finding 3 share
  one root cause — Phase-A/C eval numbers (held_ce, rank, plateau, taper
  utilization, l0 trajectory, Mimir rate) were never exported to JSON; only
  Phase-D metrics were (`metrics_gen_ncm_*_final.jsonl`, added after review-1/2's
  M4). Either ship the logs or annotate SOURCES rows as "training log, not archived".
- **EN less-revised than PT:** blocker 1 and major 2 are both stale pre-revision
  phrasings that survive only in `paper_en-us.md` — the EN file did not get the
  full correction pass the PT file got (2.68→2.79 fixed everywhere except the EN
  abstract; "four rounds" not updated to the 8-round window).

## Verdict

The repo is unusually honest and most of what I checked reconciles exactly with
the artifacts — the revision already absorbed most of the two scientific reviews
and both ABNT reviews. It is not publishable as-is for a narrow reason: the
English abstract still claims the headline result in the wrong direction
("below" a 2.75 reference where the artifact says 2.79) and undercounts the
contaminated rounds (4 vs the 8 its own §3.3 enumerates). Add the traceability
gaps (cited numbers with no artifact), the falsified sanitization claim
(`~/models/` paths in five JSONs), and one malformed sample file — all cheap
fixes, but they sit exactly on the repo's value proposition of trustworthy
numbers.

verdict: BLOCK

## Disposição dos achados (orchestrator)

**Blocker 1** — RESOLVIDO: abstract EN corrigido para "essentially at the
reference generator's level (~2.75, approximate-protocol comparison)".
**Major 1** — RESOLVIDO: "four rounds" substituído por "nearly all
pretrained-tokenizer rounds" (linhas 11, 29).
**Major 2** — RESOLVIDO: SOURCES.md anotado — números vindos de logs internos
marcados "(log interno)"; rank mediano anotado como computado de `per_pos`;
`lcm_probe.json` copiado como evidência do 0,0718 c/s; trajetória l0 anotada
como janelas de treino vs held-out do jsonl.
**Major 3** — RESOLVIDO: taper reescrito — utilização 100% a custo de
reconstrução pior (0,654 vs 0,152 do codebook base); direção explicitada.
**Major 4** — RESOLVIDO: `~/models/` sanitizado para `models/` nos 5 artefatos.
**Minors** — corrigidos os verificáveis/impactantes: ~69M/~479M, fronteiras
87,5–89,8%, k5=36,28%, ~×3,1, LI Pingzhi, internals/SHAs do review-2
sanitizados. Aceitos com nota: representative_samples.json (JSON válido,
formato repr); hardware ausente em 2 benches (campo não emitido na época);
degeneração do AR no corpus misto (consistente com a limitação de corpus já
declarada); "~2.75" com caveat de protocolo já declarado.
