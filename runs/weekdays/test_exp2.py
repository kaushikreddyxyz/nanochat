"""Plain-assert CPU tests for weekday-geometry Experiment 2 (trainable direction).

Covers exactly the three contracts the run depends on:
  [A] 7-channel weekday SUBSET slicing is correct on a synthetic climbmix-scored
      memmap fixture — the right column indices (47..53, name-sorted store order)
      pulled from the right layer axis (gemma L8 == store axis-1 index 1) of an
      int8 [n, 3, 54] scores array, dequantized+standardized exactly.
  [B] the realism THRESHOLD zeroes every post-alignment row whose max over the 7
      weekday channels is < present_z (=2.0) EXACTLY, and passes rows >= 2.0
      through untouched (boundary inclusive); None passes through as None.
  [C] runs/weekdays/exp2_config.json PARSES and round-trips through InjectionCfg,
      with the pinned exp-2 fields (r=7, after_block=3, trainable, orthonormal,
      gate abs:0.0273) and the pinned weekday channel order.

Standalone:  python runs/weekdays/test_exp2.py     (exit 0 = all pass)
Or:          python -m pytest runs/weekdays/test_exp2.py
"""
import json
import os
import sys
import tempfile
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from nanochat.injection.sites import InjectionCfg, classify_gate_spec  # noqa: E402
import nanochat.injection.sources as sources_mod  # noqa: E402
from weekday_source import WeekdayProbeScoreSource, WEEKDAY_CONCEPTS  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic climbmix-scored store: 54 concepts with the 7 weekdays at cols
# 47..53 (name-sorted store order), layers [6,8,14]. scale=1 / zero=0 / mean=0 /
# std=1 so dequant+standardize is the identity => stored int8 == returned z.
# --------------------------------------------------------------------------- #
LAYERS = [6, 8, 14]
CONCEPTS = [f"c{i:02d}" for i in range(47)] + WEEKDAY_CONCEPTS   # weekdays at 47..53
assert len(CONCEPTS) == 54
WEEKDAY_IDX = list(range(47, 54))
K = 54
NDOC = 5  # gemma tokens in the single fixture doc


def _build_store():
    d = tempfile.mkdtemp(prefix="weekday_store_")
    json.dump({"concepts": CONCEPTS, "layers": LAYERS,
               "families": {c: "weekdays" if c in WEEKDAY_CONCEPTS else "misc" for c in CONCEPTS}},
              open(os.path.join(d, "columns.json"), "w"))
    json.dump({"zero": [[0.0] * K for _ in LAYERS], "scale": [[1.0] * K for _ in LAYERS]},
              open(os.path.join(d, "quant.json"), "w"))
    json.dump({"mean": [[0.0] * K for _ in LAYERS], "std": [[1.0] * K for _ in LAYERS]},
              open(os.path.join(d, "corpus_stats.json"), "w"))
    # scores[n, layer_axis, concept]: distinctive per-axis fill so a wrong layer
    # axis or wrong column subset is caught. L8 weekday cols carry the signal.
    scores = np.zeros((NDOC, 3, K), np.int8)
    # Sentinels are all > 35 so they can't collide with the weekday signal values
    # (gi*7 + j + 1 spans 1..35), letting the "wrong axis/column" check be exact.
    scores[:, 0, :] = 101  # L6 everywhere (must NOT appear)
    scores[:, 2, :] = 102  # L14 everywhere (must NOT appear)
    scores[:, 1, :] = 100  # L8 non-weekday cols (must NOT appear)
    for gi in range(NDOC):
        for j, col in enumerate(WEEKDAY_IDX):
            scores[gi, 1, col] = gi * 7 + j + 1   # L8 weekday cols: the expected signal (1..35)
    np.save(os.path.join(d, "scores_00000.npy"), scores)
    with open(os.path.join(d, "docs_00000.jsonl"), "w") as f:
        f.write(json.dumps({"doc": 0, "start": 0, "n": NDOC}) + "\n")
    return d, scores


def _resolve_source_class(class_spec):
    """Replicate DIFF 1's source-class resolution (file-path OR dotted-module
    form) so the config's 'class' string is validated end-to-end here. File paths
    resolve from the repo root (REPO), matching the launch CWD."""
    import importlib
    target, cls = class_spec.rsplit(":", 1)
    if target.endswith(".py") or "/" in target:
        import importlib.util
        p = target if os.path.isabs(target) else os.path.join(REPO, target)
        s = importlib.util.spec_from_file_location(f"_inj_src_{cls}", p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(target)
    return getattr(m, cls)


def _make_source(store):
    # A trivial nano_enc; the subset/layer/threshold tests below never invoke the
    # gemma tokenizer path (they call _gemma_z / _align_and_gather directly), so a
    # stub enc is enough to construct the source.
    class _StubEnc:
        def encode_ordinary(self, text):
            return list(range(len(text.split())))
    return WeekdayProbeScoreSource(
        store, shards=[0], layer=8, nano_enc=_StubEnc(),
        gemma_encode=lambda t: ([], []), concepts=WEEKDAY_CONCEPTS,
        noise_sigma=0.0, present_z=2.0)


def test_subset_and_layer_axis():
    """[A] col_idx == [47..53], layer axis == L8, dequant+standardize exact."""
    store, scores = _build_store()
    src = _make_source(store)
    assert src.r == 7, f"expected r=7, got {src.r}"
    assert src.col_idx.tolist() == WEEKDAY_IDX, src.col_idx.tolist()
    assert src.li == LAYERS.index(8) == 1, (src.li,)          # gemma L8 -> store axis-1 index 1
    assert src.concepts == WEEKDAY_CONCEPTS
    assert src.present_z == 2.0

    z = src._gemma_z(0, 0, NDOC)                              # (NDOC, 7) dequant+standardized
    expected = scores[0:NDOC, 1, 47:54].astype(np.float32)   # L8 weekday cols only
    assert z.shape == (NDOC, 7), z.shape
    assert np.array_equal(z, expected), "L8 weekday subset mismatch"
    # prove it is NOT reading L6 (101) or L14 (102) or non-weekday L8 (100)
    assert not np.any(z == 101) and not np.any(z == 102) and not np.any(z == 100), \
        "pulled a wrong layer axis or wrong columns"
    print("[A] subset + layer-axis + dequant  OK")


def test_realism_threshold():
    """[B] rows with max(7 channels) < 2.0 -> exact zero; >= 2.0 -> untouched."""
    store, _ = _build_store()
    src = _make_source(store)

    # Controlled post-alignment array: isolate the threshold from tokenization by
    # temporarily stubbing the parent's aligner (what super()._align_and_gather
    # resolves to). Rows chosen to probe below / at / above the 2.0 boundary.
    fake = np.array([
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # max 3.0  >= 2 -> KEEP
        [1.9, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],   # max 1.9  <  2 -> ZERO
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],   # max 2.0  == 2 -> KEEP (inclusive)
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # max 0.5  <  2 -> ZERO
        [-4.0, -3.0, 0.1, 0.0, 0.0, 0.0, 0.0], # max 0.1  <  2 -> ZERO (neg tail ignored)
    ], np.float32)

    orig = sources_mod._RuntimeProbeBase._align_and_gather
    try:
        sources_mod._RuntimeProbeBase._align_and_gather = \
            lambda self, text, n_tokens, z_gemma: fake.copy()
        out = src._align_and_gather("irrelevant", fake.shape[0], np.zeros((fake.shape[0], 7), np.float32))
        assert np.array_equal(out[0], fake[0]), "row >=2 was altered"
        assert np.array_equal(out[2], fake[2]), "boundary row ==2 was not kept"
        assert np.all(out[1] == 0.0), "sub-2 row not zeroed"
        assert np.all(out[3] == 0.0), "sub-2 row not zeroed"
        assert np.all(out[4] == 0.0), "negative-only row not zeroed"

        # None (unknown/drift) must pass through untouched.
        sources_mod._RuntimeProbeBase._align_and_gather = \
            lambda self, text, n_tokens, z_gemma: None
        assert src._align_and_gather("x", 2, np.zeros((2, 7), np.float32)) is None
    finally:
        sources_mod._RuntimeProbeBase._align_and_gather = orig
    print("[B] realism threshold (exact zero / inclusive boundary / None)  OK")


def test_config_roundtrips():
    """[C] exp2_config.json parses + round-trips through InjectionCfg."""
    spec = json.load(open(os.path.join(HERE, "exp2_config.json")))
    sites = [InjectionCfg(**d) for d in spec["sites"]]
    assert len(sites) == 1
    s = sites[0]
    assert s.name == "weekdays" and s.r == 7 and s.after_block == 3
    assert s.trainable_direction is True
    assert s.direction_init == "orthonormal" and s.direction_seed == 1337
    assert s.optim == "adamw"
    assert s.gate == "abs:0.0273"
    mode, val = classify_gate_spec(s.gate)                   # gate string must be valid
    assert mode == "abs" and abs(val - 0.0273) < 1e-12, (mode, val)

    # sources keys match site names (injection_train asserts this)
    assert set(spec["sources"]) == {c.name for c in sites}
    ssrc = spec["sources"]["weekdays"]
    assert ssrc["kind"] == "probe-scores-runtime"
    assert ssrc["layer"] == 8
    assert ssrc["concepts"] == WEEKDAY_CONCEPTS              # pinned channel order
    assert ssrc["class"].endswith(":WeekdayProbeScoreSource")
    assert ssrc["kwargs"]["present_z"] == 2.0
    assert ssrc["align_policy"] == "mean"
    assert ssrc["noise_sigma"] == 0.0

    # The config's "class" string must actually resolve to WeekdayProbeScoreSource
    # via the SAME both-forms logic DIFF 1 adds to _open_injection_source. This
    # validates the wiring end-to-end without needing the framework edit applied.
    resolved = _resolve_source_class(ssrc["class"])   # fresh module => distinct class object
    from nanochat.injection.sources import RuntimeProbeScoreSource
    assert resolved.__name__ == "WeekdayProbeScoreSource"
    assert issubclass(resolved, RuntimeProbeScoreSource)   # same parent module (cached) => True
    assert hasattr(resolved, "_align_and_gather")

    # asdict -> InjectionCfg round-trip is stable
    s2 = InjectionCfg(**asdict(s))
    assert asdict(s2) == asdict(s)
    print("[C] exp2_config.json parse + InjectionCfg round-trip  OK")


if __name__ == "__main__":
    test_subset_and_layer_axis()
    test_realism_threshold()
    test_config_roundtrips()
    print("\nALL WEEKDAY EXP-2 TESTS PASSED")
