#!/usr/bin/env python3
"""LCM (Mimir-1.6B) CPU probe — runs INSIDE the lcm-infer container.

Mounts expected:
  /models/mimir-lcm/model.pt   (weights, ro)
  /ext/lcm_repo                (cloned repo, ro; provides `lcm` package)
  /cache                       (writable; XDG_CACHE_HOME for SONAR assets)

--warm: only downloads SONAR assets (run with network once).
Default: full probe offline — embed prompt sentences -> two-tower
diffusion LCM generates concept embeddings -> SONAR decode to text.
Writes /work/lcm_probe.json.
"""
import argparse, json, os, resource, sys, time

sys.path.insert(0, "/ext/lcm_repo")


def log(*a):
    print("[probe]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true",
                    help="only download SONAR pipelines, then exit")
    ap.add_argument("--model", default="/models/mimir-lcm/model.pt")
    ap.add_argument("--max-gen", type=int, default=6,
                    help="concepts (sentences) to generate")
    ap.add_argument("--timesteps", type=int, default=40,
                    help="diffusion denoising steps per concept")
    ap.add_argument("--prompts", nargs="*", default=[
        "The cat sits on the mat.",
        "Write a Python function that computes fibonacci numbers.",
    ])
    ap.add_argument("--out", default="/work/lcm_probe.json")
    a = ap.parse_args()

    import torch
    import lcm
    lcm.setup_fairseq2()
    from sonar.inference_pipelines.text import (
        EmbeddingToTextModelPipeline, TextToEmbeddingModelPipeline)

    dev = torch.device("cpu")
    log("loading SONAR encoder/decoder (downloads on warm run)…")
    t0 = time.time()
    enc = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder", device=dev)
    dec = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_decoder", device=dev)
    log(f"SONAR ready in {time.time()-t0:.0f}s")
    if a.warm:
        log("warm done")
        return

    from lcm.models.two_tower_diffusion_lcm.builder import (
        create_two_tower_diffusion_lcm_model)
    from lcm.models.two_tower_diffusion_lcm.archs import (
        two_tower_diffusion_lcm_1_6B)
    from lcm.inference.two_tower_diffusion_lcm.generator import (
        DiffusionLCMGeneratorOptions, TwoTowerDiffusionLCMGenerator)
    from lcm.datasets.batch import EmbeddingsBatch

    log("building two_tower_diffusion_lcm_1_6B (fp32 CPU)…")
    t0 = time.time()
    model = create_two_tower_diffusion_lcm_model(
        two_tower_diffusion_lcm_1_6B(), device=dev, dtype=torch.float32)
    sd = torch.load(a.model, map_location=dev)
    model.load_state_dict(sd.get("model", sd), strict=True)
    model.eval()
    log(f"model loaded in {time.time()-t0:.0f}s")

    opts = DiffusionLCMGeneratorOptions(inference_timesteps=a.timesteps)
    gen = TwoTowerDiffusionLCMGenerator(model, opts)

    out = {"timesteps": a.timesteps, "max_gen": a.max_gen, "runs": []}
    for prompt in a.prompts:
        t0 = time.time()
        emb = enc.predict([prompt], source_lang="eng_Latn")
        emb = emb.to(device=dev, dtype=torch.float32).unsqueeze(0)  # 1×1×D
        batch = EmbeddingsBatch(emb)
        t1 = time.time()
        res = gen(batch, max_gen_len=a.max_gen)
        hyp = res.hypotheses[0][0]
        seq = hyp.seq                                # (S+T)×D embeddings
        t2 = time.time()
        texts = dec.predict(seq, target_lang="eng_Latn")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        run = {"prompt": prompt, "n_concepts": int(seq.shape[0]),
               "embed_s": t1 - t0, "gen_s": t2 - t1,
               "concepts_per_s": (seq.shape[0] - 1) / max(t2 - t1, 1e-9),
               "max_rss_gb": rss, "sentences": list(texts)}
        out["runs"].append(run)
        log(f"prompt={prompt!r} -> {seq.shape[0]-1} concepts in "
            f"{t2-t1:.1f}s ({run['concepts_per_s']:.2f} conc/s)")
        for i, s in enumerate(texts):
            log(f"  [{i}] {s!r}")
    out["max_rss_gb"] = max(r["max_rss_gb"] for r in out["runs"])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log("->", a.out)


if __name__ == "__main__":
    main()
