"""Pod-side smoke gate for the injection-on-vs-off eval suite. Run BEFORE the
long evals (RUNBOOK.md step 2); every check here needs the pod (rustbpe / CUDA /
gated gemma / HF checkpoints), which the laptop-side tests cannot exercise.

Checks (fail loudly, in dependency order):
  1. nanochat tokenizer round-trip: decode(encode_ordinary(t)) == t over every
     weekday_v1 prompt+option string and every causal prompt (the exact texts the
     evals score), plus edge/unicode samples — the CORE adapter's decode->re-encode
     acts path relies on this (drift rows degrade to zero acts; should be ~0).
  2. Day-name first tokens distinct (the causal single-forward readout needs it).
  3. load_model('trainable') on CUDA: gate_scale=0 forward is BIT-identical to the
     acts=None forward under the POD dtype (bf16) — the on/off invariant, on-device.
  4. load_model('baseline') has NO site; attach_site + _extract_direction round-trip.
  5. GemmaScorer end-to-end (gated gemma + vendored probe constants + store stats):
     score a day sentence; the day's own STORE channel must be the argmax on the
     day token and >= 2.0 (the realism threshold the injection fired on in training).

Usage:  cd <nanochat repo root> && .venv/bin/python runs/weekdays/eval/pod_smoke.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)


def main():
    import torch
    import harness
    import causal
    import causal_items as ci
    import weekday_evalset as W
    from nanochat.tokenizer import get_tokenizer

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
    assert len(set(first)) == 7, f"day first tokens collide: {first}"
    print(f"[smoke] 2. day first tokens distinct: {first}")

    # ---- 3. on/off invariant on the POD dtype (trainable arm) ---------------
    harness.set_nano_tokenizer(enc)
    model, meta = harness.load_model("trainable", device)
    bos = tok.get_bos_token_id()
    ids = [bos] + enc.encode_ordinary("Sarah was born on a Friday. The birth was on a")
    acts = np.zeros((len(ids), 7), np.float32)
    acts[len(ids) // 2, 0] = 3.0  # synthetic firing mid-sequence
    off = harness.forward_metrics(model, ids, acts[None], gate_scale=0.0)["logits"]
    vanilla = harness.forward_metrics(model, ids, acts=None)["logits"]
    on = harness.forward_metrics(model, ids, acts[None], gate_scale=1.0)["logits"]
    assert torch.equal(off, vanilla), "gate_scale=0 not bit-identical on the pod dtype"
    assert not torch.equal(on, vanilla), "gate_scale=1 with nonzero acts is a no-op?!"
    g = model.injection_sites["weekdays"].gate.detach().float().cpu()
    assert torch.equal(g, torch.tensor(0.0273)), f"gate not restored: {g}"
    direction = causal._extract_direction(model)
    del model
    print("[smoke] 3. on/off bit-identity OK on trainable (cuda/bf16); gate restored")

    # ---- 4. baseline has no site; attach_site round-trips -------------------
    base, bmeta = harness.load_model("baseline", device)
    assert getattr(base, "injection_sites", None) is None, "baseline must have NO site"
    site = harness.attach_site(base, direction)
    got = causal._extract_direction(base)
    assert np.array_equal(got, direction.astype(np.float32)), \
        "attach_site direction round-trip failed"
    del base
    print("[smoke] 4. baseline site-free; attach_site(checkpoint direction) OK")

    # ---- 5. GemmaScorer fires the right STORE channel on a day token --------
    scorer = harness.GemmaScorer(device)
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

    print("\nSMOKE PASSED — proceed with run_evals.py and causal.py (RUNBOOK.md)")


if __name__ == "__main__":
    main()
