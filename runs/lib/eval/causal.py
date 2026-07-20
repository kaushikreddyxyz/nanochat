"""Causal/counterfactual protocol, shared by every concept family: when the TEXT says
label X but the INJECTION says label Y, which does the model follow, and how does it
scale with loudness? Conditions per item: off / clean_on / dose@{.25,.5,2,4} /
cf_swap(Y near,far; onehot+empirical) / cf_dose / implant(Y on a label-free noun). Arms
= the family's real arms + their baseline+attach_site negative controls. Readout: the
label first-token logits at the answer position. The family's item bank is bound by
--items-module (runs/<family>/eval/<family>_items.py).
Run: see runs/<family>/eval/run_all.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# repo root for `nanochat.*` imports, regardless of launch CWD (same convention
# as run_evals.py; `python path/to/causal.py` puts only the script dir on sys.path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Family bindings, all supplied by bind_items_module: the item bank, its two orderings
# (STORE = activation-channel order, CALENDAR = the semantic cycle answers are read off),
# their index/arithmetic helpers, and the item keys holding the mentioned / injected
# label. Unbound until then, so the pure functions below stay family-agnostic.
ci = None
STORE_ORDER = CALENDAR_ORDER = None
store_idx = cal_idx = day_plus = completion = None
LABEL_FIELD = INJECT_FIELD = None

PRESENT_Z = 2.0          # realism threshold (matches the site's relu knee)
LOUDNESS_DIAL = 1.0      # control-arm loudness in units of the donor arm's calibrated scale
ONEHOT_Z = 3.0           # onehot pattern amplitude (a strong single-channel firing)
DOSE_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]  # correct-pattern dose-response grid
CF_DOSE_SCALES = [2.0, 4.0]                     # extra swap-amplification scales (>1)
RESULTS_DIR = None       # set from --out-dir (the family's results dir)


# --------------------------------------------------------------------------- #
# Patterns and transforms (pure; numpy in, same-kind out; NEVER mutate input).
# --------------------------------------------------------------------------- #
def onehot_pattern(day: str) -> np.ndarray:
    """r-vector: ONEHOT_Z on day's STORE channel, 0 elsewhere (r = len(STORE_ORDER))."""
    p = np.zeros(len(STORE_ORDER), np.float32)
    p[store_idx(day)] = ONEHOT_Z
    return p


def _np(a):
    """(is_torch, numpy_view) — support torch OR numpy acts without importing torch."""
    if type(a).__module__.startswith("torch"):
        return True, a.detach().cpu().numpy()
    return False, np.asarray(a)


def _restore(is_torch, ref, arr):
    """Rebuild a torch tensor like ``ref`` from numpy ``arr`` (else return arr)."""
    if is_torch:
        import torch
        return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    return arr


def swap_transform(x_channel: int, pattern, present_z: float = PRESENT_Z):
    """acts -> acts: on every row where channel ``x_channel`` is active
    (>= present_z), REPLACE the whole 7-vector with ``pattern`` (day Y). Rows
    where X is not in its positive tail are untouched. Pure."""
    pat = np.asarray(pattern, np.float32)

    def fn(acts):
        is_t, a = _np(acts)
        out = a.copy()
        mask = a[..., x_channel] >= present_z   # token axis; works for [T,7] and [B,T,7]
        out[mask] = pat
        return _restore(is_t, acts, out)
    return fn


def implant_transform(positions, pattern):
    """acts -> acts: overwrite rows ``positions`` (a token span/list) with
    ``pattern`` (day Y), regardless of current content. Pure."""
    pat = np.asarray(pattern, np.float32)
    pos = list(positions)

    def fn(acts):
        is_t, a = _np(acts)
        out = a.copy()
        if pos:
            out[..., pos, :] = pat            # token axis; works for [T,7] and [B,T,7]
        return _restore(is_t, acts, out)
    return fn


# --------------------------------------------------------------------------- #
# Condition grid + per-item transforms.
# --------------------------------------------------------------------------- #
def conditions_for(item, emp_present: bool):
    """List of condition dicts for an item. Each: label, tkey (transform key),
    scale (loudness_scale), ctype ('agree'|'swap'|'implant'), text_answer, cf_answer,
    Y (swap/implant target day or None). Forwards are deduped on (tkey, scale)."""
    conds = []
    if item["family"] == "mention":
        x, kq, ta = item[LABEL_FIELD], item["kq"], item["text_answer"]
        # correct-pattern dose-response (identity transform, loudness sweep)
        for g in DOSE_SCALES:
            label = "off" if g == 0.0 else ("clean_on" if g == 1.0 else f"dose@{g:g}")
            conds.append(dict(label=label, tkey="identity", scale=g, ctype="agree",
                              text_answer=ta, cf_answer=None, Y=None))
        # counterfactual swaps: near (X+1) and far (X+3) in calendar
        for tag, off in (("near", 1), ("far", 3)):
            y = day_plus(x, off)
            cf = day_plus(y, kq)
            conds.append(dict(label=f"cf_swap_onehot_{tag}", tkey=f"swap_onehot_{tag}",
                              scale=1.0, ctype="swap", text_answer=ta, cf_answer=cf, Y=y))
            if emp_present:
                conds.append(dict(label=f"cf_swap_emp_{tag}", tkey=f"swap_emp_{tag}",
                                  scale=1.0, ctype="swap", text_answer=ta, cf_answer=cf, Y=y))
        # amplified swap (far target): can louder injection override the text?
        yfar = day_plus(x, 3)
        cffar = day_plus(yfar, kq)
        for g in CF_DOSE_SCALES:
            conds.append(dict(label=f"cf_dose_onehot_far@{g:g}", tkey="swap_onehot_far",
                              scale=g, ctype="swap", text_answer=ta, cf_answer=cffar, Y=yfar))
    else:  # implant (label-free)
        y, kq = item[INJECT_FIELD], item["kq"]
        cf = day_plus(y, kq)
        conds.append(dict(label="implant_off", tkey="identity", scale=0.0,
                          ctype="implant", text_answer=None, cf_answer=cf, Y=y))
        conds.append(dict(label="implant_zeroacts_on", tkey="identity", scale=1.0,
                          ctype="implant", text_answer=None, cf_answer=cf, Y=y))
        for g in [1.0] + CF_DOSE_SCALES:
            conds.append(dict(label=f"implant_onehot@{g:g}", tkey="implant_onehot",
                              scale=g, ctype="implant", text_answer=None, cf_answer=cf, Y=y))
        if emp_present:
            conds.append(dict(label="implant_emp@1", tkey="implant_emp", scale=1.0,
                              ctype="implant", text_answer=None, cf_answer=cf, Y=y))
    return conds


def build_transforms(item, acts, empirical, present_z: float = PRESENT_Z):
    """tkey -> transform callable (identity -> None, the exact no-transform path).
    ``empirical`` is {day: [7]} or None."""
    t = {"identity": None}
    if item["family"] == "mention":
        xch = store_idx(item[LABEL_FIELD])
        for tag, off in (("near", 1), ("far", 3)):
            y = day_plus(item[LABEL_FIELD], off)
            t[f"swap_onehot_{tag}"] = swap_transform(xch, onehot_pattern(y), present_z)
            if empirical is not None:
                t[f"swap_emp_{tag}"] = swap_transform(xch, empirical[y], present_z)
    else:
        y = item[INJECT_FIELD]
        span = range(*item["noun_span"])
        t["implant_onehot"] = implant_transform(span, onehot_pattern(y))
        if empirical is not None:
            t["implant_emp"] = implant_transform(span, empirical[y])
    return t


def with_bos(bos, nano_ids, acts_body, span=None):
    """Prepend BOS to a (ids, acts[, token-span]) triple, keeping acts aligned to
    INPUT positions (row t = input token t; BOS row is exactly zero — the training
    convention, and what run_evals.py does for every forward). ``span`` is a
    [start, end) token span into ``nano_ids`` (implant noun) and shifts by +1.
    Pure (unit-tested); the readout position stays "last row of the logits"."""
    ids = [bos] + list(nano_ids)
    acts_body = np.asarray(acts_body, np.float32)
    acts = np.concatenate([np.zeros((1, acts_body.shape[1]), np.float32), acts_body], axis=0)
    if span is None:
        return ids, acts, None
    return ids, acts, (span[0] + 1, span[1] + 1)


def run_item_forwards(ids, acts, transforms, conds, forward_fn):
    """Dedup conditions on (tkey, scale) and call ``forward_fn(ids, acts,
    loudness_scale, transform)`` once per unique pair. Returns label -> forward
    result. Pure w.r.t. the harness (forward_fn is injected -> unit-testable)."""
    cache, out = {}, {}
    for c in conds:
        key = (c["tkey"], c["scale"])
        if key not in cache:
            cache[key] = forward_fn(ids, acts, c["scale"], transforms[c["tkey"]])
        out[c["label"]] = cache[key]
    return out, len(cache)


# --------------------------------------------------------------------------- #
# Readout math (pure; operates on the 7 calendar-ordered day logits).
# --------------------------------------------------------------------------- #
def _softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def day_logits_from(logits_last, day_first_ids):
    """(V,) logits at the answer position -> (7,) logits at each day's first
    token id, indexed in CALENDAR order."""
    return np.asarray(logits_last, np.float64)[np.asarray(day_first_ids)]


def readout(day_logits, text_answer, cf_answer):
    """Metrics from the 7 calendar-ordered day logits.
      argmax_day, p (softmax over the 7 days),
      logit_text/p_text (if text_answer), logit_cf/p_cf (if cf_answer),
      gap = logit_cf - logit_text (injection vs text; if both given),
      follow_text / follow_cf (argmax matches)."""
    dl = np.asarray(day_logits, np.float64)
    p = _softmax(dl)
    am = int(dl.argmax())
    r = {"argmax_day": CALENDAR_ORDER[am],
         "day_logits": [round(float(v), 4) for v in dl]}
    if text_answer is not None:
        ti = cal_idx(text_answer)
        r["logit_text"] = float(dl[ti]); r["p_text"] = float(p[ti])
        r["follow_text"] = (CALENDAR_ORDER[am] == text_answer)
    if cf_answer is not None:
        yi = cal_idx(cf_answer)
        r["logit_cf"] = float(dl[yi]); r["p_cf"] = float(p[yi])
        r["follow_cf"] = (CALENDAR_ORDER[am] == cf_answer)
    if text_answer is not None and cf_answer is not None:
        r["gap"] = float(dl[cal_idx(cf_answer)] - dl[cal_idx(text_answer)])
    return r


def ce_over_name(logits_seq, name_ids, answer_pos):
    """CE (sum of -log p) of teacher-forced ``name_ids`` starting at
    ``answer_pos`` in a (T,V) logits array (prompt + name forward). For the
    optional --full-name-ce readout. Pure."""
    lg = np.asarray(logits_seq, np.float64)
    total = 0.0
    for j, tid in enumerate(name_ids):
        p = _softmax(lg[answer_pos + j])
        total += -np.log(max(p[tid], 1e-12))
    return float(total)


# --------------------------------------------------------------------------- #
# Aggregation across items (pure).
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate_arm(per_item):
    """per_item: list of {item, conds:{label:readout}}. Returns condition-level
    aggregates (mean gap / flip / follow / logit_text) with by-day breakdown and
    off-baseline deltas, plus the two dose-response curves."""
    # group readouts by condition label
    by_label = {}
    off_by_item = {}      # item_id -> off readout (agree baseline)
    impoff_by_item = {}   # item_id -> implant_off readout
    for rec in per_item:
        iid = rec["item"]["id"]
        for label, ro in rec["conds"].items():
            by_label.setdefault(label, []).append((rec["item"], ro))
            if label == "off":
                off_by_item[iid] = ro
            if label == "implant_off":
                impoff_by_item[iid] = ro

    summary = {}
    for label, rows in by_label.items():
        ctype = _ctype_of(label)
        agg = {"ctype": ctype, "n": len(rows)}
        gaps = [ro.get("gap") for _, ro in rows]
        agg["mean_gap"] = _mean(gaps)
        agg["flip_rate"] = _mean([1.0 if ro.get("follow_cf") else 0.0
                                  for _, ro in rows if "follow_cf" in ro])
        agg["follow_text_rate"] = _mean([1.0 if ro.get("follow_text") else 0.0
                                         for _, ro in rows if "follow_text" in ro])
        agg["mean_logit_text"] = _mean([ro.get("logit_text") for _, ro in rows])
        agg["mean_p_cf"] = _mean([ro.get("p_cf") for _, ro in rows])
        # effect vs off baseline
        if ctype == "agree":
            agg["mean_dlogit_text_vs_off"] = _mean([
                (ro.get("logit_text") - off_by_item[it["id"]].get("logit_text"))
                for it, ro in rows if it["id"] in off_by_item
                and ro.get("logit_text") is not None])
        elif ctype == "swap":
            agg["mean_dgap_vs_off"] = _mean([
                (ro.get("gap") - off_by_item[it["id"]].get("gap"))
                for it, ro in rows if it["id"] in off_by_item
                and off_by_item[it["id"]].get("gap") is not None and ro.get("gap") is not None])
        elif ctype == "implant":
            agg["mean_dp_cf_vs_off"] = _mean([
                (ro.get("p_cf") - impoff_by_item[it["id"]].get("p_cf"))
                for it, ro in rows if it["id"] in impoff_by_item])
        # by-day breakdown (mention: by X; implant: by injected Y)
        by_day = {}
        for it, ro in rows:
            d = it.get(LABEL_FIELD) or it.get(INJECT_FIELD)
            by_day.setdefault(d, {"gap": [], "flip": [], "logit_text": []})
            by_day[d]["gap"].append(ro.get("gap"))
            if "follow_cf" in ro:
                by_day[d]["flip"].append(1.0 if ro["follow_cf"] else 0.0)
            by_day[d]["logit_text"].append(ro.get("logit_text"))
        agg["by_day"] = {d: {"mean_gap": _mean(v["gap"]), "flip_rate": _mean(v["flip"]),
                             "mean_logit_text": _mean(v["logit_text"])}
                         for d, v in by_day.items()}
        summary[label] = agg
    return summary


def _ctype_of(label):
    if label.startswith("implant"):
        return "implant"
    if label.startswith("cf_"):
        return "swap"
    return "agree"


# --------------------------------------------------------------------------- #
# Runtime (harness). Imported lazily so the pure module + tests need no harness.
# --------------------------------------------------------------------------- #
def _logits_last(out):
    """Extract the answer-position (last) logits as numpy (V,) from whatever
    forward_metrics returns (tensor / dict['logits'] / obj.logits)."""
    lg = out
    if isinstance(out, dict):
        lg = out.get("logits", out.get("logit"))
    elif hasattr(out, "logits"):
        lg = out.logits
    if lg is None:
        raise RuntimeError("forward_metrics returned no logits; causal.py needs "
                           "per-position logits (see HARNESS CONTRACT).")
    if type(lg).__module__.startswith("torch"):
        lg = lg.detach().float().cpu().numpy()
    lg = np.asarray(lg, np.float64)
    while lg.ndim > 2:      # (B,T,V) -> drop batch
        lg = lg[0]
    if lg.ndim == 2:        # (T,V) -> answer position is the last row
        lg = lg[-1]
    return lg               # (V,)


def _site_params(model, site_name):
    """The arm's actual site — direction (learned for trainable, frozen for sphere/
    orthogonal) plus the calibrated channel_scale/threshold/after_block — straight from
    its loaded checkpoint, which is the faithful thing to bolt onto the baseline
    negative control."""
    import harness
    return harness.site_params(model, site_name)


def resolve_arms(spec, real_arms=("trainable", "sphere", "orthogonal")):
    """'all' -> every real arm + its baseline control; else a comma list. Each:
    (label, base_arm, control) where control means baseline+attach_site(direction of
    base_arm). ``real_arms`` are the family's real checkpoint FOLDER names."""
    real = list(real_arms)
    ctrl = [(f"baseline_{a}", a, True) for a in real]
    allcfg = [(a, a, False) for a in real] + ctrl
    if spec == "all":
        return allcfg
    want = [s.strip() for s in spec.split(",")]
    return [c for c in allcfg if c[0] in want]


def bind_items_module(path):
    """Point the driver at a family's item bank (file path -> module) and bind the
    family globals the pure functions read. Loaded by path under a fixed module name so
    two families' banks can never collide in sys.modules. Required before run()."""
    global ci, STORE_ORDER, CALENDAR_ORDER, store_idx, cal_idx, day_plus, completion
    global LABEL_FIELD, INJECT_FIELD
    import importlib.util
    spec = importlib.util.spec_from_file_location("_causal_items_family", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    ci = m
    STORE_ORDER = list(m.STORE_ORDER)
    CALENDAR_ORDER = list(getattr(m, "CALENDAR_ORDER", None) or m.CYCLE_ORDER)
    store_idx = m.store_idx
    cal_idx = getattr(m, "cal_idx", None) or m.cycle_idx
    day_plus = getattr(m, "day_plus", None) or m.season_plus
    completion = m.completion
    LABEL_FIELD = getattr(m, "LABEL_FIELD", "day")
    INJECT_FIELD = getattr(m, "INJECT_FIELD", "inject_day")
    return m


def run(args):
    global RESULTS_DIR
    import harness                              # sibling harness (landed)
    from nanochat.tokenizer import get_tokenizer  # rustbpe present on the training pod

    bind_items_module(args.items_module)        # STORE/CALENDAR/idx/plus + label fields
    RESULTS_DIR = args.out_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    concepts = harness.family_concepts(args.family)
    # One canonical channel order: the GemmaScorer concepts MUST equal the item bank's
    # STORE order, or every counterfactual is silently mislabeled.
    assert list(concepts) == list(STORE_ORDER), \
        f"scorer concepts {list(concepts)} != STORE_ORDER {list(STORE_ORDER)}"

    tok = get_tokenizer()
    enc = tok.enc                               # tiktoken Encoding (encode_ordinary ...)
    bos = tok.get_bos_token_id()
    harness.set_nano_tokenizer(enc)             # build_acts uses the injected tokenizer

    empirical = _load_empirical(args.empirical_json)
    items = ci.generate_items()
    if args.limit:
        items = items[:args.limit]
    print(f"[causal] {len(items)} items; empirical={'yes' if empirical else 'no'}")

    day_first_ids, _name_ids = _day_token_ids(enc)
    scorer = harness.GemmaScorer(args.device, concepts, layer=args.layer)
    prepared = _prepare_items(items, enc, scorer, harness, bos)

    lm_kw = {"hf_repo": args.hf_repo, "step": args.step}
    site_cache = {}
    all_summaries, n_fwd_total = {}, 0
    for label, base_arm, control in resolve_arms(args.arms, real_arms=args.real_arms):
        if control:                             # baseline + the arm's own site (never trained w/ it)
            model, _meta = harness.load_model("baseline", args.device, **lm_kw)
            harness.attach_site(model, name=args.site_name, dial=args.loudness,
                                **_site_params_for(base_arm, harness, args.device,
                                                   site_cache, args.site_name, lm_kw))
        else:
            model, _meta = harness.load_model(base_arm, args.device, **lm_kw)
            site_cache[base_arm] = _site_params(model, args.site_name)   # for its control
        per_item, n_fwd = _eval_arm(harness, model, prepared, day_first_ids, empirical, args)
        n_fwd_total += n_fwd
        summ = aggregate_arm(per_item)
        all_summaries[label] = summ
        _write_arm(label, base_arm, control, per_item, summ, empirical is not None)
        print(f"[causal] arm {label}: {n_fwd:,} forwards -> results/causal_{label}.json")
        del model

    _write_summary(all_summaries, empirical is not None, n_fwd_total, len(items))
    print(f"[causal] {n_fwd_total:,} total forwards -> results/causal_summary.json")


def _load_empirical(path):
    if not os.path.exists(path):
        print(f"[causal] no empirical patterns at {path} — onehot conditions only "
              f"(run empirical_patterns.py to add them)")
        return None
    with open(path) as f:
        ej = json.load(f)
    assert ej["store_order"] == STORE_ORDER, "empirical store_order != STORE_ORDER"
    print(f"[causal] empirical patterns loaded ({path})")
    return {d: np.asarray(v, np.float32) for d, v in ej["vectors"].items()}


def _site_params_for(arm, harness, device, cache, site_name, lm_kw):
    """Cached arm site params; loads the arm checkpoint to extract them if the real arm
    wasn't evaluated this run (real arms are processed first, so the cache is
    normally already warm)."""
    if arm not in cache:
        amodel, _ = harness.load_model(arm, device, **lm_kw)
        cache[arm] = _site_params(amodel, site_name)
        del amodel
    return cache[arm]


def _day_token_ids(enc):
    """First-token ids (CANONICAL order) for the ' <label>' completions + full name
    ids. Asserts all first tokens are DISTINCT (the single-forward readout needs it);
    else fail loudly (the labels would need the CE-over-full-name path)."""
    name_ids = [enc.encode_ordinary(completion(d)) for d in CALENDAR_ORDER]
    first = [ids[0] for ids in name_ids]
    if len(set(first)) != len(CALENDAR_ORDER):
        raise RuntimeError(f"label first tokens not distinct: {first} "
                           f"(names={[completion(d) for d in CALENDAR_ORDER]}); "
                           f"use ce_over_name (--full-name-ce)")
    return np.asarray(first), name_ids


def _prepare_items(items, enc, scorer, harness, bos):
    """Attach per-item (ids, acts[, noun_span]). ids = [BOS] + body tokens with a
    zero BOS acts row (the training/run_evals convention — every suite forwards
    BOS-prefixed sequences; the answer readout is still the LAST logits row).
    acts is arm-independent:
      * mention: real gemma weekday z (scorer.score_with_offsets) aligned +
        thresholded by harness.build_acts (exactly the training path);
      * implant: day-free by construction, so acts are EXACTLY zero and the
        transform supplies the whole injected signal (no scorer call)."""
    prepared = []
    for it in items:
        nano_ids = enc.encode_ordinary(it["prompt"])
        T = len(nano_ids)
        if it["family"] == "mention":
            gz, off = scorer.score_with_offsets([it["prompt"]])[0]
            acts = np.asarray(harness.build_acts(it["prompt"], nano_ids, gz, off,
                              threshold=PRESENT_Z, policy="max", nano_enc=enc), np.float32)
            r = len(STORE_ORDER)
            assert acts.shape == (T, r), f"build_acts {acts.shape} != {(T, r)} for {it['id']}"
            ids, acts, _ = with_bos(bos, nano_ids, acts)
            prepared.append(dict(it, ids=ids, acts=acts))
        else:
            span = ci.noun_token_span(enc.encode_ordinary, it["prompt"], *it["noun_char"])
            ids, acts, span = with_bos(bos, nano_ids, np.zeros((T, len(STORE_ORDER)), np.float32), span)
            prepared.append(dict(it, ids=ids, acts=acts, noun_span=span))
    return prepared


def _eval_arm(harness, model, prepared, day_first_ids, empirical, args):
    def forward_fn(ids, acts, loudness_scale, transform):
        out = harness.forward_metrics(model, ids, acts, loudness_scale=loudness_scale,
                                      transform=transform)
        return _logits_last(out)

    per_item, n_fwd = [], 0
    for it in prepared:
        conds = conditions_for(it, empirical is not None)
        transforms = build_transforms(it, it["acts"], empirical)
        raw, nf = run_item_forwards(it["ids"], it["acts"], transforms, conds, forward_fn)
        n_fwd += nf
        cread = {}
        for c in conds:
            dl = day_logits_from(raw[c["label"]], day_first_ids)
            cread[c["label"]] = readout(dl, c["text_answer"], c["cf_answer"])
        per_item.append({"item": {k: it[k] for k in it
                                  if k not in ("ids", "acts")}, "conds": cread})
    return per_item, n_fwd


def _write_arm(label, base_arm, control, per_item, summ, emp):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"causal_{label}.json"), "w") as f:
        json.dump({"arm": label, "base_arm": base_arm, "negative_control": control,
                   "empirical": emp, "present_z": PRESENT_Z, "loudness_dial": LOUDNESS_DIAL,
                   "onehot_z": ONEHOT_Z, "n_items": len(per_item),
                   "summary": summ, "items": per_item}, f, indent=1)


def _write_summary(all_summaries, emp, n_fwd, n_items):
    arms = list(all_summaries)
    labels = sorted({l for s in all_summaries.values() for l in s})
    conditions = {lab: {arm: all_summaries[arm].get(lab) for arm in arms
                        if lab in all_summaries[arm]} for lab in labels}
    # dose-response matrices
    correct = {}   # arm -> loudness scale -> mean dlogit_text vs off
    counterf = {}  # arm -> loudness scale -> {flip_rate, mean_gap} for the swapped-far dose
    for arm, s in all_summaries.items():
        cm = {}
        for g in DOSE_SCALES:
            lab = "off" if g == 0.0 else ("clean_on" if g == 1.0 else f"dose@{g:g}")
            if lab in s:
                cm[str(g)] = s[lab].get("mean_dlogit_text_vs_off")
        correct[arm] = cm
        km = {}
        for g in [1.0] + CF_DOSE_SCALES:
            lab = "cf_swap_onehot_far" if g == 1.0 else f"cf_dose_onehot_far@{g:g}"
            if lab in s:
                km[str(g)] = {"flip_rate": s[lab].get("flip_rate"),
                              "mean_gap": s[lab].get("mean_gap")}
        counterf[arm] = km
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "causal_summary.json"), "w") as f:
        json.dump({"arms": arms, "conditions": conditions,
                   "dose_response": {"correct_vs_off": correct,
                                     "counterfactual_far": counterf},
                   "meta": {"empirical_present": emp, "forward_passes": n_fwd,
                            "n_items": n_items, "present_z": PRESENT_Z,
                            "loudness_dial": LOUDNESS_DIAL, "store_order": STORE_ORDER,
                            "calendar_order": CALENDAR_ORDER}}, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description="causal/counterfactual eval (family-parameterized)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="all",
                    help="'all' (every real arm + its baseline control) or a comma list of labels")
    ap.add_argument("--real-arms", nargs="+", required=True,
                    help="the family's real checkpoint FOLDER names")
    ap.add_argument("--items-module", required=True,
                    help="path to the family's item bank (runs/<family>/eval/<family>_items.py)")
    ap.add_argument("--family", required=True,
                    help="concept family in runs/lib/concepts.py (supplies the probe concepts "
                         "in STORE order)")
    ap.add_argument("--site-name", required=True, help="injection site name in the checkpoints")
    ap.add_argument("--hf-repo", required=True, help="checkpoint HF repo for every arm")
    ap.add_argument("--step", type=int, required=True, help="checkpoint step")
    ap.add_argument("--loudness", type=float, default=LOUDNESS_DIAL,
                    help="dial on the baseline-control site's loudness, in units of the donor "
                         "arm's own calibrated channel_scale (1.0 = as loud as that arm). The "
                         "arm's threshold and after_block come from its checkpoint too.")
    ap.add_argument("--layer", type=int, default=8, help="gemma layer for the probes")
    ap.add_argument("--out-dir", required=True, help="the family's results dir")
    ap.add_argument("--empirical-json", default="",
                    help="the family's empirical_patterns.json; absent = onehot conditions only")
    ap.add_argument("--limit", type=int, default=0, help="cap item count (smoke)")
    ap.add_argument("--full-name-ce", action="store_true",
                    help="(reserved) CE-over-full-name readout; default = first-token logits")
    args = ap.parse_args()
    if args.full_name_ce:
        raise NotImplementedError("--full-name-ce: use ce_over_name(); the default "
                                  "first-token readout suffices when label first tokens "
                                  "are distinct (asserted at startup).")
    run(args)


if __name__ == "__main__":
    main()
