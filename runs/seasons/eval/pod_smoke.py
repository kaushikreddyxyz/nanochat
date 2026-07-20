"""Pod-side smoke gate — run BEFORE the long evals (run_all.sh step 1); needs rustbpe/
CUDA/gated gemma/HF checkpoints. Verifies tokenizer round-trip on the exact eval texts,
season first-token distinctness, on-device loudness-0 bit-identity (the on/off invariant),
baseline-has-no-site, and the gemma probes firing the right STORE channel. Uses the shared
runs/lib/eval harness. Run: .venv/bin/python runs/seasons/eval/pod_smoke.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "runs", "lib", "eval"))  # shared harness

HF_REPO = "kaushikreddyxyz/nanochat-d12-injections"
STEP = 2520
LAYER = 8
FAMILY = "seasons"
SITE_NAME = "seasons"
THRESHOLD = 2.0
TRAINABLE_ARM = "seasons_trainable_L0"


def main():
    import torch
    import seasons_items as ci
    import seasons_evalset as S
    from seasons_concepts import CYCLE_ORDER, STORE_ORDER, R

    import harness
    from nanochat.tokenizer import get_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    # ---- 1. tokenizer round-trip on the actual eval texts --------------------
    tok = get_tokenizer()
    enc = tok.enc
    texts = []
    for it in S.generate_items():
        texts += [it["prompt"] + " " + o for o in it["options"]]
    for it in ci.generate_items():
        texts.append(it["prompt"])
    texts += ["Hello, world!", "naïve café — “smart quotes” … 3×4=12",
              "tabs\tand\nnewlines", "emoji 🙂 and CJK 日本語", "  leading spaces"]
    bad = 0
    for t in texts:
        if tok.decode(enc.encode_ordinary(t)) != t:
            bad += 1
            if bad <= 5:
                print(f"[smoke] ROUNDTRIP FAIL: {t!r}")
    assert bad == 0, f"{bad}/{len(texts)} texts fail decode(encode(t))==t"
    print(f"[smoke] 1. tokenizer round-trip OK on {len(texts)} texts")

    # ---- 2. season first tokens distinct ------------------------------------
    first = [enc.encode_ordinary(ci.completion(s))[0] for s in CYCLE_ORDER]
    assert len(set(first)) == R, f"season first tokens collide: {first}"
    print(f"[smoke] 2. season first tokens distinct: {first}")

    # ---- 3. on/off invariant on the POD dtype (trainable arm) ---------------
    harness.set_nano_tokenizer(enc)
    model, meta = harness.load_model(TRAINABLE_ARM, device, HF_REPO, STEP)
    site_name = (meta.get("site_name") if isinstance(meta, dict) else None) or SITE_NAME
    bos = tok.get_bos_token_id()
    ids = [bos] + enc.encode_ordinary("Sarah was born in autumn. The birth was in")
    acts = np.zeros((len(ids), R), np.float32)
    acts[len(ids) // 2, 0] = 3.0  # synthetic firing mid-sequence
    off = harness.forward_metrics(model, ids, acts[None], loudness_scale=0.0)["logits"]
    vanilla = harness.forward_metrics(model, ids, acts=None)["logits"]
    on = harness.forward_metrics(model, ids, acts[None], loudness_scale=1.0)["logits"]
    assert torch.equal(off, vanilla), "loudness_scale=0 not bit-identical on the pod dtype"
    assert not torch.equal(on, vanilla), "loudness_scale=1 with nonzero acts is a no-op?!"
    site = model.injection_sites[site_name]
    assert site.cfg.r == R, f"site r={site.cfg.r} != {R}"
    print(f"[smoke] 3. on/off bit-identity OK on trainable; site {site_name!r} "
          f"after_block={site.cfg.after_block}")
    del model

    # ---- 4. baseline has no site -------------------------------------------
    base, _ = harness.load_model("baseline", device, HF_REPO, STEP)
    assert getattr(base, "injection_sites", None) is None, "baseline must have NO site"
    del base
    print("[smoke] 4. baseline site-free")

    # ---- 5. GemmaScorer fires the right STORE channel on a season word ------
    scorer = harness.GemmaScorer(device, harness.family_concepts(FAMILY), layer=LAYER)
    season = "winter"
    text = f"The village is always quiet in {season} because of the snow."
    z, offsets = scorer.score_with_offsets([text])[0]
    lo = text.index(season)
    hi = lo + len(season)
    rows = [i for i, (s, e) in enumerate(np.asarray(offsets)) if s < hi and e > lo]
    assert rows, "no gemma token covers the season word?!"
    zseason = np.max(np.asarray(z)[rows], axis=0)
    ch = int(np.argmax(zseason))
    assert STORE_ORDER[ch] == season, \
        f"argmax channel {STORE_ORDER[ch]} != {season} (z={zseason.round(2).tolist()})"
    assert zseason[ch] >= THRESHOLD, \
        f"season z {zseason[ch]:.2f} below present_z={THRESHOLD}"
    print(f"[smoke] 5. GemmaScorer OK: {season} channel z={zseason[ch]:.2f} (store {STORE_ORDER})")

    print("\nSMOKE PASSED — proceed with the shared run_evals (see run_all.sh)")


if __name__ == "__main__":
    main()
