"""CPU tests for weekday_evalset + the shared run_evals' pure reductions:
byte-determinism, committed-jsonl freshness, day balance, leak checks, scoring math on
stub logits, bucket masks, summary schema. No torch/rustbpe.
Run: python -m pytest runs/tests/test_weekday_evalset.py
"""
import math
import os

import numpy as np

import run_evals as R  # must import WITHOUT torch
import weekday_evalset as W
from conftest import WEEKDAYS

HERE = os.path.join(WEEKDAYS, "eval")


# --------------------------------------------------------------------------- #
# Generation: determinism, freshness, balance, leaks, option well-formedness.
# --------------------------------------------------------------------------- #
def test_deterministic_bytes():
    b1 = W.to_jsonl_bytes(W.generate_items())
    b2 = W.to_jsonl_bytes(W.generate_items())
    assert b1 == b2, "generation is not byte-deterministic"


def test_committed_jsonl_is_fresh():
    path = os.path.join(HERE, "evalsets", "weekday_v1.jsonl")
    assert os.path.exists(path), f"missing generated eval set: {path}"
    with open(path, "rb") as f:
        on_disk = f.read()
    assert on_disk == W.to_jsonl_bytes(W.generate_items()), \
        "committed weekday_v1.jsonl is stale — re-run weekday_evalset.py"


def test_unique_ids_and_size():
    items = W.generate_items()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    assert 400 <= len(items) <= 600, f"item count {len(items)} outside ~400-600"


def test_balanced_day_counts():
    items = W.generate_items()
    from collections import Counter
    c = Counter(it["answer"] for it in items if it["category"] in W.BALANCED_DAY_CATEGORIES)
    assert set(c) == set(W.WEEKDAYS), f"balanced categories miss some days: {set(c)}"
    assert len(set(c.values())) == 1, f"per-day answer counts not balanced: {dict(c)}"


def test_no_answer_leak_in_fill_in():
    items = W.generate_items()
    leaks = [it["id"] for it in items if it["meta"]["fill_in"] and W.leaks_answer(it)]
    assert not leaks, f"fill-in items leak the answer into the prompt: {leaks}"


def test_options_well_formed():
    items = W.generate_items()
    for it in items:
        assert it["answer"] in it["options"], (it["id"], "answer not in options")
        assert len(it["options"]) == len(set(it["options"])), (it["id"], "duplicate options")
        assert len(it["options"]) >= 2, (it["id"], "need >=2 options")
        assert not it["prompt"].endswith(" "), (it["id"], "prompt must not end with a space")
        # day-answer categories present the full week
        if it["category"] in ("order_next", "order_prev", "order_k", "fact_position"):
            assert it["options"] == W.WEEKDAYS, (it["id"], "expected the 7 weekdays as options")


def test_categories_have_multiple_templates():
    items = W.generate_items()
    from collections import defaultdict
    tmpls = defaultdict(set)
    for it in items:
        tmpls[it["category"]].add(it["meta"].get("template"))
    for cat in ("order_next", "order_prev", "order_k", "fact_position", "usage_context"):
        assert len(tmpls[cat]) >= 3, f"{cat} has <3 surface templates ({len(tmpls[cat])})"


# --------------------------------------------------------------------------- #
# Scoring math on a stub model (no torch / no real tokenizer).
# --------------------------------------------------------------------------- #
class StubTokenizer:
    """Whitespace word-level tokenizer. Each unique word -> a stable id; BOS=0.
    Day/sanity options are single words, so continuations are one token each."""

    def __init__(self):
        self.vocab = {"<bos>": 0}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None):
        ids = [self._id(w) for w in text.split()]
        if prepend is not None:
            ids = [prepend] + ids
        return ids


def per_token_ce_from_logits(logits, ids):
    """Harness convention: CE[t] = -log_softmax(logits[t-1])[ids[t]] for t>=1
    (score the PREDICTED token t); position 0 = nan (no context)."""
    logits = np.asarray(logits, dtype=np.float64)
    ce = np.full(len(ids), np.nan)
    for t in range(1, len(ids)):
        row = logits[t - 1]
        logZ = np.log(np.exp(row - row.max()).sum()) + row.max()
        ce[t] = logZ - row[ids[t]]
    return ce


def _score_items(items, model_fn, tok):
    """Return (accuracy, per_item_option_ces) using the run_evals reductions."""
    correct, per_item = 0, []
    for it in items:
        options = it["options"]
        answer_index = options.index(it["answer"])
        seqs = [tok.encode(it["prompt"] + " " + o, prepend=tok.get_bos_token_id()) for o in options]
        start = max(1, R.common_prefix_len(seqs))
        opt_ce = []
        for ids in seqs:
            logits = model_fn(it, ids)
            ce = per_token_ce_from_logits(logits, ids)
            opt_ce.append(R.option_mean_ce(ce, start, len(ids)))
        pred, _ = R.predict_from_option_ces(opt_ce)
        correct += int(pred == answer_index)
        per_item.append(opt_ce)
    return correct / len(items), per_item


def _sample_items():
    # a small, single-token-option slice keeps the stub trivial + fast
    items = [it for it in W.generate_items()
             if it["category"] in ("order_next", "order_prev", "sanity")]
    return items[:40]


def test_uniform_logits_are_chance():
    tok = StubTokenizer()
    items = _sample_items()
    # fix a vocab size by pre-tokenizing everything through the stub
    for it in items:
        for o in it["options"]:
            tok.encode(it["prompt"] + " " + o, prepend=0)
    V = len(tok.vocab)

    def uniform(_it, ids):
        return np.zeros((len(ids), V))

    _acc, per_item = _score_items(items, uniform, tok)
    # a uniform model cannot distinguish options: all option CEs equal (margin 0).
    for ces in per_item:
        assert max(ces) - min(ces) < 1e-9, f"uniform model gave unequal option CEs: {ces}"


def test_rigged_logits_are_perfect():
    tok = StubTokenizer()
    items = _sample_items()
    for it in items:
        for o in it["options"]:
            tok.encode(it["prompt"] + " " + o, prepend=0)
    V = len(tok.vocab)

    def rigged(it, ids):
        # favor the GOLD continuation token at every position, regardless of which
        # option is being scored -> the answer option gets ~0 CE, others high.
        gold_id = tok.encode(it["answer"])[0]
        logits = np.zeros((len(ids), V))
        logits[:, gold_id] = 30.0
        return logits

    acc, _ = _score_items(items, rigged, tok)
    assert acc == 1.0, f"rigged (oracle) model should be perfect, got {acc}"


# --------------------------------------------------------------------------- #
# run_evals: CLI parse, bucket masks, summary schema.
# --------------------------------------------------------------------------- #
_REQUIRED = ["--family", "weekdays", "--evalset", "x.jsonl",
             "--hf-repo", "r", "--step", "2520", "--out-dir", "out"]


def test_cli_parses():
    p = R.build_parser()
    a = p.parse_args(_REQUIRED + ["--arms", "baseline", "trainable",
                                  "--metrics", "completion", "valbpb",
                                  "--injection", "on", "off",
                                  "--heldout-shards", "100", "101",
                                  "--valbpb-source", "gemma", "--core-max-per-task", "500"])
    assert a.arms == ["baseline", "trainable"]
    assert a.metrics == ["completion", "valbpb"]
    assert a.injection == ["on", "off"]
    assert a.heldout_shards == [100, 101]
    assert a.valbpb_source == "gemma"
    assert a.core_max_per_task == 500
    # defaults
    d = R.build_parser().parse_args(_REQUIRED + ["--arms", "baseline"])
    assert d.injection == ["on", "off"] and d.site_name is None
    assert d.metrics == ["completion", "valbpb", "core"]
    assert d.heldout_shards == R.HELDOUT_SHARDS_DEFAULT


def test_scored_repo_mapping():
    assert R.scored_repo_for_shard(0) == "kaushikreddyxyz/climbmix-scored"
    assert R.scored_repo_for_shard(24) == "kaushikreddyxyz/climbmix-scored"
    assert R.scored_repo_for_shard(25) == "kaushikreddyxyz/climbmix-scored-overflow"
    assert R.scored_repo_for_shard(100) == "kaushikreddyxyz/climbmix-scored-overflow-4"
    assert R.scored_repo_for_shard(184) == "kaushikreddyxyz/climbmix-scored-overflow-7"


def test_bucket_masks_partition():
    # acts nonzero at INPUT token positions 1 and 3. Buckets are over PREDICTED
    # tokens t (harness.ce_report convention): injected = acts[t] nonzero;
    # after = acts[t-1] nonzero & not injected; position 0 is invalid (BOS/NaN).
    acts = [[0, 0], [1, 0], [0, 0], [0, 3], [0, 0], [0, 0]]
    valid = [False] + [True] * 5  # position 0 has no predicted-token CE
    m = R.bucket_masks(acts, valid)
    T = len(acts)
    for t in range(T):  # every valid predicted position is in exactly one bucket
        hits = sum(int(m[b][t]) for b in ("injected", "after", "rest"))
        assert hits == (1 if valid[t] else 0), (t, {b: m[b][t] for b in m})
    assert m["injected"][1] and m["injected"][3]
    assert m["after"][2] and m["after"][4]
    assert m["rest"][5]
    assert not (m["injected"][0] or m["after"][0] or m["rest"][0])


def test_bpb_math():
    assert R.bpb_from_nats_bytes(0.0, 10) == 0.0
    assert R.bpb_from_nats_bytes(5.0, 0) == float("inf")
    assert abs(R.bpb_from_nats_bytes(math.log(2) * 10, 10) - 1.0) < 1e-9


def test_summary_merge_schema():
    recs = [
        {"metric": "weekdays_accuracy", "arm": "trainable", "injection": "on",
         "value": 0.42, "n": 422},
        {"metric": "weekdays_accuracy", "arm": "trainable", "injection": "off",
         "value": 0.30, "n": 422},
        {"metric": "valbpb_injected", "arm": "trainable", "injection": "on",
         "value": 0.9, "n": 1234},
        {"metric": "core_metric", "arm": "baseline", "injection": "off",
         "value": 0.14, "n": 22},
    ]
    s = R.merge_summary(recs)
    assert s["weekdays_accuracy"]["trainable"]["on"] == {"value": 0.42, "n": 422}
    assert s["weekdays_accuracy"]["trainable"]["off"]["value"] == 0.30
    assert s["valbpb_injected"]["trainable"]["on"]["n"] == 1234
    assert s["core_metric"]["baseline"]["off"]["value"] == 0.14
    # schema shape: metric -> arm -> injection -> {value, n}
    for metric, arms in s.items():
        for arm, cells in arms.items():
            for injection, cell in cells.items():
                assert injection in ("on", "off")
                assert set(cell) == {"value", "n"}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
