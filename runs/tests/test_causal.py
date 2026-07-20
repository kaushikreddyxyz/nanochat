"""CPU tests for the SHARED causal protocol's pure parts, bound to the weekday item
bank: store/calendar mapping, transforms, condition grid, forward dedup, readout math
(torch only to prove transforms are array-library agnostic).
Run: python -m pytest runs/tests/test_causal.py
"""
import os

import numpy as np

import causal
import weekday_items as ci
from conftest import WEEKDAYS
from weekday_items import (CALENDAR_ORDER, STORE_ORDER, cal_idx, day_plus,
                           noun_token_span, store_idx)

causal.bind_items_module(os.path.join(WEEKDAYS, "eval", "weekday_items.py"))


# --------------------------------------------------------------------------- #
# 1. Store-vs-calendar mapping (the classic trap) — asserted for all 7 days.
# --------------------------------------------------------------------------- #
def test_store_is_name_sorted():
    assert STORE_ORDER == sorted(STORE_ORDER)              # climbmix name-sorted cols 47..53
    assert STORE_ORDER == ["friday", "monday", "saturday", "sunday",
                           "thursday", "tuesday", "wednesday"]
    assert CALENDAR_ORDER == ["monday", "tuesday", "wednesday", "thursday",
                              "friday", "saturday", "sunday"]


def test_mapping_all_days():
    for d in STORE_ORDER:
        assert 0 <= store_idx(d) < 7 and STORE_ORDER[store_idx(d)] == d
        assert 0 <= cal_idx(d) < 7 and CALENDAR_ORDER[cal_idx(d)] == d
    # store index != calendar index for most days -> the trap is real
    assert store_idx("friday") == 0 and cal_idx("friday") == 4
    assert store_idx("monday") == 1 and cal_idx("monday") == 0


def test_day_plus():
    assert day_plus("sunday", 1) == "monday"      # wrap
    assert day_plus("monday", -1) == "sunday"     # wrap back
    assert day_plus("friday", 3) == "monday"      # fri->sat->sun->mon
    assert day_plus("tuesday", 1) == "wednesday"
    assert day_plus("tuesday", 2) == "thursday"
    for d in CALENDAR_ORDER:
        assert day_plus(d, 7) == d and day_plus(d, 0) == d


# --------------------------------------------------------------------------- #
# 2. Patterns + transforms (onehot exact; swap hits exactly active-X; purity;
#    implant positions; torch/numpy parity).
# --------------------------------------------------------------------------- #
def test_onehot_pattern():
    p = causal.onehot_pattern("monday")
    assert p.shape == (7,) and p[store_idx("monday")] == causal.ONEHOT_Z
    assert p.sum() == causal.ONEHOT_Z                     # only one channel set


def _acts_fixture():
    a = np.zeros((5, 7), np.float32)
    a[1, store_idx("friday")] = 2.5    # active on X=friday
    a[3, store_idx("friday")] = 3.1    # active on X=friday
    a[0, store_idx("friday")] = 1.0    # below present_z -> not active
    a[2, store_idx("thursday")] = 2.2  # active but on a different channel
    return a


def test_swap_hits_exactly_active_x_and_is_pure():
    a = _acts_fixture()
    a0 = a.copy()
    pat = causal.onehot_pattern("monday")
    fn = causal.swap_transform(store_idx("friday"), pat, present_z=2.0)
    out = fn(a)
    assert np.allclose(out[1], pat) and np.allclose(out[3], pat)   # exactly the active-X rows
    assert np.allclose(out[0], a0[0])                              # below threshold: untouched
    assert np.allclose(out[2], a0[2]) and np.allclose(out[4], a0[4])
    assert np.allclose(a, a0), "swap_transform mutated its input (must be pure)"


def test_swap_empirical_pattern():
    a = _acts_fixture()
    emp = np.array([0.1, 2.9, 0.2, 0.0, 0.4, 0.0, 0.0], np.float32)  # a Y=monday-ish vector
    fn = causal.swap_transform(store_idx("friday"), emp, present_z=2.0)
    out = fn(a)
    assert np.allclose(out[1], emp) and np.allclose(out[3], emp)


def test_swap_torch_numpy_parity():
    import torch
    a = _acts_fixture()
    pat = causal.onehot_pattern("saturday")
    fn = causal.swap_transform(store_idx("friday"), pat, present_z=2.0)
    out_np = fn(a)
    ta = torch.tensor(a)
    out_t = fn(ta)
    assert isinstance(out_t, torch.Tensor)
    assert np.allclose(out_t.numpy(), out_np)
    assert torch.allclose(ta, torch.tensor(a)), "torch input mutated (must be pure)"


def test_implant_transform():
    a = np.zeros((6, 7), np.float32)
    a0 = a.copy()
    pat = causal.onehot_pattern("wednesday")
    fn = causal.implant_transform(range(2, 4), pat)   # inject on tokens 2,3
    out = fn(a)
    assert np.allclose(out[2], pat) and np.allclose(out[3], pat)
    for r in (0, 1, 4, 5):
        assert np.allclose(out[r], 0.0)
    assert np.allclose(a, a0), "implant_transform mutated its input"


def test_noun_token_span():
    # whitespace stub tokenizer: token count == word count. The implant nouns are
    # space-delimited so prefix-length gives the exact span.
    enc = lambda s: s.split()
    prompt = "Mark was born in a small town. When asked which day, the answer is a"
    cs = len("Mark was born in a small")           # start of " town"
    ce = cs + len(" town")
    start, end = noun_token_span(enc, prompt, cs, ce)
    assert (start, end) == (6, 7)                    # "town" is word index 6
    assert prompt.split()[start] == "town."          # (stub keeps trailing punct)


# --------------------------------------------------------------------------- #
# 3. Condition grid + dose wiring + forward dedup.
# --------------------------------------------------------------------------- #
def _mention_item():
    its = [i for i in ci.generate_items() if i["family"] == "mention"]
    return next(i for i in its if i["day"] == "friday" and i["kind"] == "recall")


def test_conditions_dose_grid_and_targets():
    it = _mention_item()                              # X=friday, kq=0, text_answer=friday
    conds = causal.conditions_for(it, emp_present=False)
    scales = {c["scale"] for c in conds if c["tkey"] == "identity"}
    assert scales == set(causal.DOSE_SCALES)          # full correct-pattern dose sweep
    labels = {c["label"] for c in conds}
    assert {"off", "clean_on", "cf_swap_onehot_near", "cf_swap_onehot_far",
            "cf_dose_onehot_far@2", "cf_dose_onehot_far@4"} <= labels
    # swap targets: near=X+1, far=X+3; recall kq=0 so cf_answer == Y
    near = next(c for c in conds if c["label"] == "cf_swap_onehot_near")
    far = next(c for c in conds if c["label"] == "cf_swap_onehot_far")
    assert near["Y"] == day_plus("friday", 1) == "saturday" and near["cf_answer"] == "saturday"
    assert far["Y"] == day_plus("friday", 3) == "monday" and far["cf_answer"] == "monday"


def test_empirical_conditions_gated_on_presence():
    it = _mention_item()
    assert not any("emp" in c["label"] for c in causal.conditions_for(it, False))
    assert any("emp" in c["label"] for c in causal.conditions_for(it, True))


def test_derived_counterfactual_answer():
    its = [i for i in ci.generate_items() if i["family"] == "mention"]
    it = next(i for i in its if i["kind"] == "derived" and i["template"] == "D1"
              and i["day"] == "tuesday")              # kq=+1 -> text_answer=wednesday
    assert it["text_answer"] == "wednesday"
    conds = causal.conditions_for(it, False)
    far = next(c for c in conds if c["label"] == "cf_swap_onehot_far")
    # Y = tuesday+3 = friday; cf answer = Y + kq(+1) = saturday
    assert far["Y"] == "friday" and far["cf_answer"] == "saturday"


def test_run_item_forwards_dedup_and_loudness_wiring():
    it = _mention_item()
    acts = np.zeros((4, 7), np.float32)
    acts[2, store_idx("friday")] = 2.5
    conds = causal.conditions_for(it, emp_present=False)
    transforms = causal.build_transforms(it, acts, empirical=None)
    calls = []

    def fake_forward(ids, a, loudness_scale, transform):
        calls.append((loudness_scale, transform is None))
        # return a fake (T,V) logits array; identity => vector of zeros
        return np.zeros((len(ids), 32))

    raw, n_unique = causal.run_item_forwards([0, 1, 2, 3], acts, transforms,
                                             conds, fake_forward)
    # every condition got a result; forwards were deduped on (tkey, scale)
    assert set(raw) == {c["label"] for c in conds}
    assert n_unique == len({(c["tkey"], c["scale"]) for c in conds})
    assert len(calls) == n_unique
    # the identity/off forward used loudness scale 0 with transform=None
    assert (0.0, True) in calls
    # a swap forward passed a real transform at loudness scale 1
    assert (1.0, False) in calls


def test_implant_conditions():
    it = next(i for i in ci.generate_items() if i["family"] == "implant"
              and i["inject_day"] == "wednesday")
    it = dict(it, noun_span=(6, 7))
    conds = causal.conditions_for(it, emp_present=True)
    labels = {c["label"] for c in conds}
    assert {"implant_off", "implant_zeroacts_on", "implant_onehot@1",
            "implant_onehot@2", "implant_onehot@4", "implant_emp@1"} == labels
    assert all(c["ctype"] == "implant" for c in conds)
    assert all(c["cf_answer"] == "wednesday" for c in conds)  # kq=0 -> cf==Y
    t = causal.build_transforms(it, np.zeros((8, 7), np.float32),
                                empirical={d: np.ones(7, np.float32) for d in CALENDAR_ORDER})
    assert t["identity"] is None and callable(t["implant_onehot"])


def test_with_bos():
    """BOS prepend keeps acts aligned to input positions (zero BOS row) and
    shifts the implant noun span by exactly +1 — the run_evals/training
    convention causal forwards now share."""
    body = np.arange(21, dtype=np.float32).reshape(3, 7)
    ids, acts, span = causal.with_bos(5, [10, 11, 12], body, (1, 3))
    assert ids == [5, 10, 11, 12]
    assert acts.shape == (4, 7)
    assert np.all(acts[0] == 0.0), "BOS acts row must be exactly zero"
    assert np.array_equal(acts[1:], body)
    assert span == (2, 4)
    ids2, acts2, span2 = causal.with_bos(5, [10], np.zeros((1, 7), np.float32))
    assert ids2 == [5, 10] and span2 is None and acts2.shape == (2, 7)


# --------------------------------------------------------------------------- #
# 4. Readout math on rigged logits (constructed flip + agree cases).
# --------------------------------------------------------------------------- #
def test_day_logits_from_indexing():
    V = 50
    logits = np.full(V, -9.0)
    day_first_ids = np.arange(10, 17)                 # calendar day d -> id 10+cal_idx(d)
    for d in CALENDAR_ORDER:
        logits[10 + cal_idx(d)] = float(cal_idx(d))   # ascending by calendar index
    dl = causal.day_logits_from(logits, day_first_ids)
    assert list(dl) == [0, 1, 2, 3, 4, 5, 6]


def test_readout_flip_case():
    # text says friday, injection (cf) says monday; rig monday highest.
    dl = np.full(7, 0.0)
    dl[cal_idx("monday")] = 5.0
    dl[cal_idx("friday")] = 1.0
    r = causal.readout(dl, text_answer="friday", cf_answer="monday")
    assert r["argmax_day"] == "monday"
    assert r["follow_cf"] is True and r["follow_text"] is False
    assert r["gap"] == 5.0 - 1.0                        # logit_cf - logit_text
    assert r["logit_cf"] == 5.0 and r["logit_text"] == 1.0
    assert r["p_cf"] > r["p_text"]


def test_readout_agree_case():
    # correct-pattern condition: text=tuesday, no counterfactual; rig tuesday top.
    dl = np.full(7, 0.0)
    dl[cal_idx("tuesday")] = 3.0
    r = causal.readout(dl, text_answer="tuesday", cf_answer=None)
    assert r["follow_text"] is True and "gap" not in r and "follow_cf" not in r
    assert r["logit_text"] == 3.0


def test_ce_over_name():
    # rigged (T,V) logits: at answer_pos put all mass on token 7, next pos on 8.
    V = 16
    lg = np.full((3, V), -20.0)
    lg[1, 7] = 20.0
    lg[2, 8] = 20.0
    ce = causal.ce_over_name(lg, name_ids=[7, 8], answer_pos=1)
    assert ce < 1e-3                                    # near-perfect -> CE ~ 0
    ce_bad = causal.ce_over_name(lg, name_ids=[0, 0], answer_pos=1)
    assert ce_bad > 10.0                                # wrong tokens -> large CE


# --------------------------------------------------------------------------- #
# 5. Aggregation sanity (pure).
# --------------------------------------------------------------------------- #
def test_aggregate_arm_basic():
    it = {"id": "m1", "family": "mention", "day": "friday", "kind": "recall"}
    per_item = [{
        "item": it,
        "conds": {
            "off": {"gap": 0.0, "follow_text": True, "follow_cf": False,
                    "logit_text": 1.0, "p_cf": 0.1, "day_logits": [0] * 7},
            "cf_swap_onehot_far": {"gap": 2.0, "follow_text": False, "follow_cf": True,
                                   "logit_text": 0.5, "p_cf": 0.6, "day_logits": [0] * 7},
        }}]
    summ = causal.aggregate_arm(per_item)
    assert summ["cf_swap_onehot_far"]["flip_rate"] == 1.0
    assert summ["cf_swap_onehot_far"]["mean_gap"] == 2.0
    assert summ["cf_swap_onehot_far"]["mean_dgap_vs_off"] == 2.0    # 2.0 - 0.0
    assert "friday" in summ["cf_swap_onehot_far"]["by_day"]


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
