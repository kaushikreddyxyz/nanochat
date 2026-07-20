"""Pod-side smoke gate — run BEFORE the long evals (run_all.sh step 2); needs rustbpe/
CUDA/gated gemma/HF checkpoints. Verifies tokenizer round-trip on the exact eval texts,
day-first-token distinctness, on-device loudness-0 bit-identity (the on/off invariant),
baseline-has-no-site + attach_site round-trip, and GemmaScorer end-to-end. If this
fails, stop — nothing downstream is meaningful.
Run from repo root: .venv/bin/python runs/weekdays/eval/pod_smoke.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "runs", "lib", "eval"))   # the shared harness
sys.path.insert(0, HERE)

HF_REPO = "kaushikreddyxyz/weekday-geometry-d12"
STEP = 2520
FAMILY = "weekdays"
SITE_NAME = "weekdays"
TRAINABLE_ARM = "trainable"


def main():
    import torch
    import harness
    import causal
    import weekday_items as ci
    import weekday_evalset as W
    from nanochat.tokenizer import get_tokenizer

    causal.bind_items_module(os.path.join(HERE, "weekday_items.py"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    # ---- 1. tokenizer round-trip on the actual eval texts -------------------
    tok = get_tokenizer()
    enc = tok.enc
    texts = []
    for it in W.generate_items():
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

    # ---- 2. day-name first tokens distinct ----------------------------------
    first = [enc.encode_ordinary(ci.completion(d))[0] for d in ci.CALENDAR_ORDER]
    assert len(set(first)) == len(ci.CALENDAR_ORDER), f"day first tokens collide: {first}"
    print(f"[smoke] 2. day first tokens distinct: {first}")

    # ---- 3. on/off invariant on the POD dtype (trainable arm) ---------------
    harness.set_nano_tokenizer(enc)
    model, meta = harness.load_model(TRAINABLE_ARM, device, HF_REPO, STEP)
    bos = tok.get_bos_token_id()
    ids = [bos] + enc.encode_ordinary("Sarah was born on a Friday. The birth was on a")
    r = len(ci.STORE_ORDER)
    acts = np.zeros((len(ids), r), np.float32)
    acts[len(ids) // 2, 0] = 3.0  # synthetic firing mid-sequence
    off = harness.forward_metrics(model, ids, acts[None], loudness_scale=0.0)["logits"]
    vanilla = harness.forward_metrics(model, ids, acts=None)["logits"]
    on = harness.forward_metrics(model, ids, acts[None], loudness_scale=1.0)["logits"]
    assert torch.equal(off, vanilla), "loudness_scale=0 not bit-identical on the pod dtype"
    assert not torch.equal(on, vanilla), "loudness_scale=1 with nonzero acts is a no-op?!"
    params = harness.site_params(model, SITE_NAME)
    cs = model.injection_sites[SITE_NAME].channel_scale.detach().float().cpu().numpy()
    assert np.array_equal(cs, params["channel_scale"]), f"channel_scale not restored: {cs}"
    del model
    print("[smoke] 3. on/off bit-identity OK on trainable (cuda/bf16); loudness restored")

    # ---- 4. baseline has no site; attach_site round-trips -------------------
    base, bmeta = harness.load_model("baseline", device, HF_REPO, STEP)
    assert getattr(base, "injection_sites", None) is None, "baseline must have NO site"
    site = harness.attach_site(base, name=SITE_NAME, **params)
    got = harness.site_params(base, SITE_NAME)
    assert np.array_equal(got["direction"], params["direction"]), \
        "attach_site direction round-trip failed"
    assert np.array_equal(got["channel_scale"], params["channel_scale"]), \
        "attach_site must carry the arm's calibrated loudness"
    del base
    print("[smoke] 4. baseline site-free; attach_site(arm direction + loudness) OK")

    # ---- 5. GemmaScorer fires the right STORE channel on a day token --------
    scorer = harness.GemmaScorer(device, harness.family_concepts(FAMILY))
    day = "wednesday"
    text = f"The committee meets every {day.capitalize()} at noon."
    z, offsets = scorer.score_with_offsets([text])[0]
    lo = text.index(day.capitalize())
    hi = lo + len(day)
    rows = [i for i, (s, e) in enumerate(np.asarray(offsets)) if s < hi and e > lo]
    assert rows, "no gemma token covers the day word?!"
    zday = np.max(np.asarray(z)[rows], axis=0)
    ch = int(np.argmax(zday))
    assert ci.STORE_ORDER[ch] == day, \
        f"argmax channel {ci.STORE_ORDER[ch]} != {day} (z={zday.round(2).tolist()})"
    assert zday[ch] >= 2.0, f"day z {zday[ch]:.2f} below the present_z=2.0 threshold"
    print(f"[smoke] 5. GemmaScorer OK: {day} channel z={zday[ch]:.2f} "
          f"(store order {json.dumps(ci.STORE_ORDER)})")

    print("\nSMOKE PASSED — proceed with run_evals.py and causal.py (see run_all.sh)")


if __name__ == "__main__":
    main()
