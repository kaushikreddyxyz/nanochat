"""CPU, dependency-light tests for the weekday-geometry OPEN-GENERATION battery.

Plain asserts, no pytest required (`python runs/weekdays/eval/test_opengen.py`),
also collectible by pytest. No harness / gemma / real model / rustbpe: the acts
construction, day detection, seeding, delta math and summary schema are pure, and
the generation LOOP is exercised end-to-end against a STUB forward_fn + stub
tokenizer (torch is used only as the array/sampling backend, same as the model).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_gen as og
import opengen_items as oi
from causal_items import CALENDAR_ORDER


# --------------------------------------------------------------------------- #
# 1. Item set: 3 families, ~30 prompts, none containing a weekday.
# --------------------------------------------------------------------------- #
def test_items_day_free_and_families():
    its = oi.generate_items()
    assert 25 <= len(its) <= 40
    fams = {f for f in oi.families()}
    assert {i["family"] for i in its} == fams
    for it in its:                                    # no prompt may name a weekday
        low = it["prompt"].lower()
        assert not any(d in low for d in oi.DAY_NAMES), it["id"]
    # the {item} sweep: one shared template SD1 over >= 6 distinct nouns
    sd1 = [i for i in its if i["template"] == "SD1"]
    assert len(sd1) >= 6 and len({i["item"] for i in sd1}) == len(sd1)


# --------------------------------------------------------------------------- #
# 2. Acts construction per condition, at each generation step — EXACT positions.
#    BOS row (index 0) is ALWAYS zero. Lb = prompt-body count; final prompt token
#    at index Lb; frontier at index T-1.
# --------------------------------------------------------------------------- #
_P = np.array([1, 2, 3, 4, 5, 6, 7], np.float32)


def test_acts_none_is_none():
    assert og.acts_for_step("none", 3, 4, _P) is None


def test_acts_inject_all_positions():
    Lb = 3
    for step in range(4):
        T = 1 + Lb + step
        a = og.acts_for_step("inject_all", Lb, T, _P)
        assert a.shape == (T, 7)
        assert np.all(a[0] == 0.0), "BOS row must be zero"
        assert np.all(a[1:] == _P), "every non-BOS position gets the pattern"


def test_acts_inject_last_is_static():
    Lb = 3
    for step in range(4):
        T = 1 + Lb + step
        a = og.acts_for_step("inject_last", Lb, T, _P)
        assert np.array_equal(a[Lb], _P), "final prompt token always injected"
        others = [r for r in range(T) if r != Lb]
        assert np.all(a[others] == 0.0), "only the final prompt token; generated rows zero"


def test_acts_inject_frontier_moves():
    Lb = 3
    for step in range(4):
        T = 1 + Lb + step
        a = og.acts_for_step("inject_frontier", Lb, T, _P)
        assert np.array_equal(a[T - 1], _P), "current last position injected"
        assert np.all(a[:T - 1] == 0.0), "nothing behind the frontier"
        assert np.all(a[0] == 0.0)


def test_last_vs_frontier_coincide_step0_diverge_after():
    Lb = 3
    T0 = 1 + Lb                                        # step 0
    assert np.array_equal(og.acts_for_step("inject_last", Lb, T0, _P),
                          og.acts_for_step("inject_frontier", Lb, T0, _P))
    T1 = 2 + Lb                                        # step 1
    last1 = og.acts_for_step("inject_last", Lb, T1, _P)
    front1 = og.acts_for_step("inject_frontier", Lb, T1, _P)
    assert np.array_equal(last1[Lb], _P) and np.all(last1[Lb + 1] == 0.0)
    assert np.all(front1[Lb] == 0.0) and np.array_equal(front1[T1 - 1], _P)
    assert not np.array_equal(last1, front1)


# --------------------------------------------------------------------------- #
# 3. Day-name detection: word boundaries, multi-word decode, case, first-wins.
# --------------------------------------------------------------------------- #
def test_first_day_mentioned():
    assert og.first_day_mentioned(" on Monday")[0] == "monday"
    assert og.first_day_mentioned("next  Wednesday please")[0] == "wednesday"
    assert og.first_day_mentioned("TUESDAY")[0] == "tuesday"            # case-insensitive
    assert og.first_day_mentioned("the weekend")[0] is None
    assert og.first_day_mentioned("")[0] is None
    assert og.first_day_mentioned("Monday then Friday")[0] == "monday"  # earliest wins
    assert og.first_day_mentioned("Fridays are great")[0] is None       # strict word boundary
    assert og.first_day_mentioned("it was sunny")[0] is None            # no false substring


def test_day_distribution_and_rates():
    days = ["monday", "monday", None, "friday", None]
    D = og.day_distribution(days)
    assert D["monday"] == 2 and D["friday"] == 1 and D["none"] == 2 and D["_n"] == 5
    assert abs(og.mention_rate(D) - 3 / 5) < 1e-9


# --------------------------------------------------------------------------- #
# 4. Seeding determinism.
# --------------------------------------------------------------------------- #
def test_seed_for_determinism():
    a = og.seed_for("trainable", "personal_fact/PF1", "monday", "inject_all", "onehot", 4.0)
    b = og.seed_for("trainable", "personal_fact/PF1", "monday", "inject_all", "onehot", 4.0)
    c = og.seed_for("trainable", "personal_fact/PF1", "tuesday", "inject_all", "onehot", 4.0)
    assert a == b and a != c
    assert 0 <= a < 2 ** 31


# --------------------------------------------------------------------------- #
# 5. Delta-vs-none math.
# --------------------------------------------------------------------------- #
def test_combo_effect_vs_none():
    none = og.day_distribution(["friday"] * 8 + ["monday"] * 2 + [None] * 6)   # n=16
    combo = ["monday"] * 10 + [None] * 6                                       # Y=monday
    eff = og.combo_effect(combo, "monday", none)
    assert abs(eff["p_first_eq_Y"] - 10 / 16) < 1e-9
    assert abs(eff["d_p_first_eq_Y"] - (10 / 16 - 2 / 16)) < 1e-9              # vs none's monday
    assert abs(eff["day_mention_rate"] - 10 / 16) < 1e-9
    assert abs(eff["d_mention"] - (10 / 16 - 10 / 16)) < 1e-9                  # both mention 10/16


# --------------------------------------------------------------------------- #
# 6. Logit readout + summary schema (natural-first, deltas, wiring verdict).
# --------------------------------------------------------------------------- #
def test_logit_readout():
    dl = [0.0] * 7
    dl[CALENDAR_ORDER.index("friday")] = 5.0
    r = og.logit_readout(dl, Y="friday")
    assert r["argmax_day"] == "friday" and r["argmax_eq_Y"] is True
    assert r["p_Y"] > 1 / 7
    flat = og.logit_readout([1.0] * 7, Y="monday")
    assert abs(flat["p_Y"] - 1 / 7) < 1e-9 and flat["argmax_eq_Y"] is (flat["argmax_day"] == "monday")


def _mk_combo(cond, pattern, dose, Y, sample_days, day_logits):
    return dict(condition=cond, pattern=pattern, dose=dose, Y=Y,
                sample_days=sample_days, greedy_day=(sample_days[0] if sample_days else None),
                greedy_text="", day_logits=day_logits)


def test_summarize_arm_natural_and_injection():
    # one template; none says friday-heavy; strongest injection flips generation to Y
    # but the first-token logit stays flat -> the wiring-verification "mis-wired" branch.
    none = {"sample_days": ["friday"] * 8 + [None] * 8, "greedy_text": "friday",
            "greedy_day": "friday", "day_logits": [1.0] * 7}
    combos = [
        _mk_combo("inject_all", "onehot", 4.0, "monday", ["monday"] * 16, [1.0] * 7),
        _mk_combo("inject_last", "onehot", 1.0, "monday", ["friday"] * 8 + [None] * 8, [1.0] * 7),
    ]
    arm = {"arm": "trainable", "kind": "real", "has_site": True,
           "templates": [{"id": "t1", "family": "personal_fact", "template": "PF1",
                          "item": None, "prompt": "x", "none": none, "combos": combos}]}
    s = og.summarize_arm(arm)
    nat = s["natural_propensity"]
    assert nat["per_day_base_rate"]["friday"] == 0.5 and nat["day_mention_rate"] == 0.5
    key = "inject_all/onehot/@4"
    inj = s["injection"][key]
    assert abs(inj["mean_p_first_eq_Y"] - 1.0) < 1e-9
    assert abs(inj["mean_d_p_first_eq_Y_vs_none"] - 1.0) < 1e-9            # none said monday 0/16
    assert inj["by_Y"]["monday"]["mean_p_first_eq_Y"] == 1.0
    v = s["verification"]
    assert v["generation_moves"] is True and v["logit_readout_moves"] is False
    assert "mis-wired" in v["verdict"]


def test_summarize_arm_no_site():
    none = {"sample_days": ["monday"] * 4 + [None] * 12, "greedy_text": "", "greedy_day": None,
            "day_logits": [0.0] * 7}
    arm = {"arm": "baseline", "kind": "plain", "has_site": False,
           "templates": [{"id": "t1", "family": "schedule", "template": "SD2",
                          "item": None, "prompt": "x", "none": none}]}
    s = og.summarize_arm(arm)
    assert "injection" not in s and "verification" not in s
    assert s["natural_propensity"]["per_day_base_rate"]["monday"] == 0.25


def test_build_summary_schema():
    arm_summaries = {
        "trainable": og.summarize_arm({
            "arm": "trainable", "kind": "real", "has_site": True,
            "templates": [{"id": "t1", "family": "personal_fact", "template": "PF1",
                           "item": None, "prompt": "x",
                           "none": {"sample_days": [None] * 16, "greedy_text": "",
                                    "greedy_day": None, "day_logits": [0.0] * 7},
                           "combos": [_mk_combo("inject_all", "onehot", 4.0, "monday",
                                                ["monday"] * 16, [0.0] * 7)]}]}),
        "baseline": og.summarize_arm({
            "arm": "baseline", "kind": "plain", "has_site": False,
            "templates": [{"id": "t1", "family": "personal_fact", "template": "PF1",
                           "item": None, "prompt": "x",
                           "none": {"sample_days": [None] * 16, "greedy_text": "",
                                    "greedy_day": None, "day_logits": [0.0] * 7}}]}),
    }
    summ = og.build_summary(arm_summaries, meta={"n_prompts": 1})
    assert set(summ) == {"natural_propensity_FIRST", "injection_effects_vs_none",
                         "wiring_verification", "meta"}
    assert "trainable" in summ["natural_propensity_FIRST"]
    assert "baseline" in summ["natural_propensity_FIRST"]
    assert summ["injection_effects_vs_none"]["trainable"] is not None
    assert "baseline" not in summ["injection_effects_vs_none"]           # no-site arm omitted


# --------------------------------------------------------------------------- #
# 7. Generation loop end-to-end against a STUB forward_fn + stub tokenizer.
# --------------------------------------------------------------------------- #
_ID2WORD = {3: "on", 5: "monday", 6: "and", 7: "friday"}


def _decode(ids):
    return " ".join(_ID2WORD.get(int(i), "x") for i in ids)


class _StubForward:
    """Records the acts it is handed each step; emits a fixed token schedule for
    every batch row (one token given an overwhelming logit, so greedy AND every
    sampled row pick it -> deterministic completion for assertions)."""
    def __init__(self, V, schedule, Lb):
        self.V, self.schedule, self.Lb, self.calls = V, schedule, Lb, []

    def __call__(self, ids, acts_batch, gate_scale):
        import torch
        B, T = ids.shape
        step = T - (1 + self.Lb)
        self.calls.append((step, None if acts_batch is None else np.array(acts_batch),
                           float(gate_scale)))
        tgt = self.schedule[step] if step < len(self.schedule) else 0
        logits = torch.zeros((B, self.V), dtype=torch.float32)
        logits[:, tgt] = 100.0
        return logits


def test_generation_loop_detects_day_and_reads_logits():
    Lb, V = 3, 16
    ids0 = [99, 100, 101, 102]                          # BOS + 3 body tokens (values ignored)
    day_first_ids = np.array([5, 7, 1, 2, 4, 8, 9])     # calendar-ordered stub ids
    fwd = _StubForward(V, schedule=[3, 5, 6, 7], Lb=Lb)  # -> "on monday and friday"
    res = og.generate_completions(
        fwd, _decode, ids0, Lb, "inject_all", _P, 2.0, day_first_ids,
        n_samples=4, max_new=4, temperature=0.8, top_k=50, seed=123, device="cpu")
    assert res["greedy_day"] == "monday"                # "on monday and friday" -> first is monday
    assert res["sample_days"] == ["monday"] * 4         # forced token => all rows agree
    assert res["day_logits"] == [0.0] * 7               # none of the day ids hit the +100 token
    # gate_scale threaded through; step-0 acts match acts_for_step exactly
    assert fwd.calls[0][2] == 2.0
    a0 = fwd.calls[0][1]                                 # (B, 1+Lb, 7)
    assert a0.shape == (5, 1 + Lb, 7)
    assert np.array_equal(a0[0], og.acts_for_step("inject_all", Lb, 1 + Lb, _P))


def test_generation_none_passes_no_acts():
    Lb, V = 2, 12
    fwd = _StubForward(V, schedule=[3], Lb=Lb)
    res = og.generate_completions(
        fwd, _decode, [99, 100, 101], Lb, "none", None, 1.0, np.arange(7),
        n_samples=2, max_new=3, temperature=0.8, top_k=50, seed=7, device="cpu")
    assert all(c[1] is None for c in fwd.calls), "none condition must inject no acts"
    assert res["greedy_day"] is None                    # 'on x x' has no weekday


def test_generation_seed_reproducible():
    # a genuinely stochastic stub (two equal-logit tokens) -> seed controls the draw
    class _Stoch:
        def __init__(s): s.calls = 0
        def __call__(s, ids, acts, g):
            import torch
            s.calls += 1
            lg = torch.zeros((ids.shape[0], 8), dtype=torch.float32)
            lg[:, 5] = 1.0; lg[:, 7] = 1.0             # monday vs friday, equal
            return lg
    kw = dict(Lb=2, condition="inject_all", pattern=_P, gate_scale=1.0,
              day_first_ids=np.arange(7), n_samples=8, max_new=2,
              temperature=1.0, top_k=50, device="cpu")
    r1 = og.generate_completions(_Stoch(), _decode, [99, 1, 2], seed=42, **kw)
    r2 = og.generate_completions(_Stoch(), _decode, [99, 1, 2], seed=42, **kw)
    r3 = og.generate_completions(_Stoch(), _decode, [99, 1, 2], seed=99, **kw)
    assert r1["sample_days"] == r2["sample_days"]        # same seed -> identical draws
    assert r1["sample_days"] != r3["sample_days"]        # different seed -> different (w.h.p.)


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
