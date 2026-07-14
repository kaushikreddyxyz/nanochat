"""OPEN-GENERATION battery for the weekday-geometry injection study.

The causal/counterfactual eval (causal.py) read a single answer-position logit.
This follow-up instead LETS THE MODEL GENERATE and asks: with a day-Y pattern
injected, does a weekday actually surface in free text, which day, and how does
that compare to the model's NATURAL propensity (no injection)? Every prompt
(opengen_items.py) is DAY-FREE, so clean activations are exactly zero and any day
that appears is either the prior or the injection — never a copied prompt word.

Three injection geometries (the user's three asks), each a day-Y pattern placed
differently as generation proceeds:
  * ``inject_all``      — pattern on EVERY prompt token AND every newly generated
    token (the moving frontier included).
  * ``inject_last``     — pattern on ONLY the final prompt token, a STATIC single
    position; zero everywhere else, including generated tokens.
  * ``inject_frontier`` — pattern only on the CURRENT last position at each step
    (moves with the frontier). Coincides with ``inject_last`` at step 0, diverges
    after (that is the whole distinction; both are reported separately).
Plus ``none`` — NO injection at all, the natural-propensity reference for every
delta. Doses k in {1,2,4} multiply the trained gate; onehot (z=3.0, primary) at
all doses, the empirical cross-channel pattern at k=1. Y sweeps all 7 days.

Arms: trainable / sphere / orthogonal (site in their checkpoints), their 3
``baseline_<arm>`` untrained controls (baseline + attach_site of the arm's
checkpoint direction), and the plain ``baseline`` (NO site — pure natural
propensity, injection literally impossible). Only the 6 sited arms run the
injection grid; every arm runs ``none``.

Readouts per generation:
  1. does a weekday appear in the first ``max_new`` tokens (word-boundary,
     case-insensitive), and which day FIRST — greedy + N sampled completions.
  2. the 7-day first-token logit distribution at the first generation position
     (calendar-ordered) — the SAME mechanism the old implant readout used, kept
     here as a comparability + WIRING check (§ verification): if generation
     shifts under injection but this logit readout stays flat (~1/7), the old
     implant readout was mis-wired.

Everything below the RUNTIME banner needs torch + harness + rustbpe (pod only);
everything above is pure (numpy/stdlib) and unit-tested on CPU with a stub model
(test_opengen.py). The generation loop takes an INJECTED forward_fn, so the loop,
the acts construction, day detection and sampling are all exercised without a
real model. Pattern builders and empirical loading are REUSED from causal.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import causal                                     # pure at import (numpy only)
import causal_items as ci
import opengen_items as oi
from causal_items import CALENDAR_ORDER, STORE_ORDER

# Injection geometries and the dose/pattern grid (pinned).
CONDITIONS = ["inject_all", "inject_last", "inject_frontier"]
ONEHOT_DOSES = [1.0, 2.0, 4.0]     # gate multipliers for the onehot pattern
EMP_DOSES = [1.0]                  # empirical cross-channel pattern only at k=1
GATE_TRAINED = causal.GATE_TRAINED  # 0.0273 abs; gate_scale k => effective k*gate
ONEHOT_Z = causal.ONEHOT_Z          # 3.0 (irrelevant after the site's z/rms(z) norm)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The strongest condition — the one predicted most likely to move generation —
# used for the wiring-verification comparison (generation-moves vs logit-moves).
STRONGEST = ("inject_all", "onehot", 4.0)
MOVE_EPS = 0.03                     # a rate/prob shift above this counts as "moved"


# --------------------------------------------------------------------------- #
# 1. Acts construction per condition, per generation step (PURE).
#    Convention (harness / with_bos): acts row t aligns to INPUT token t; the BOS
#    row (index 0) is ALWAYS exactly zero. ids = [BOS] + <Lb prompt-body tokens>
#    + <generated so far>, so the final prompt token sits at index Lb and the
#    current frontier at index T-1.
# --------------------------------------------------------------------------- #
def acts_for_step(condition, Lb, T, pattern):
    """(T, r) float32 acts for this step, or None for ``none`` (no injection).
    ``pattern`` is the day-Y r-vector. BOS row 0 is never injected.
      inject_all      -> pattern on rows 1..T-1 (every non-BOS position).
      inject_last     -> pattern on row Lb only (final prompt token; FIXED).
      inject_frontier -> pattern on row T-1 only (current last position; MOVES).
    """
    if condition == "none":
        return None
    P = np.asarray(pattern, np.float32).reshape(-1)
    acts = np.zeros((T, P.shape[0]), np.float32)
    if condition == "inject_all":
        acts[1:] = P
    elif condition == "inject_last":
        assert 1 <= Lb < T, f"inject_last: final-prompt index {Lb} out of range for T={T}"
        acts[Lb] = P
    elif condition == "inject_frontier":
        acts[T - 1] = P
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return acts


# --------------------------------------------------------------------------- #
# 2. Day-name detection in generated text (PURE).
# --------------------------------------------------------------------------- #
_DAY_RE = re.compile(r"\b(" + "|".join(oi.DAY_NAMES) + r")\b", re.IGNORECASE)


def first_day_mentioned(text):
    """(day_lower | None, char_start). Word-boundary, case-insensitive; the
    EARLIEST day name in the string wins (multi-token names decode to the full
    word, so this matches ' Wednesday' but not 'Wed' or 'Fridays')."""
    m = _DAY_RE.search(text or "")
    return (m.group(1).lower(), m.start()) if m else (None, -1)


def day_distribution(first_days):
    """list of (day|None) -> {7 calendar days: count} + {'none': count} +
    {'_n': total}. The absolute first-mention distribution for a set of samples."""
    dist = {d: 0 for d in CALENDAR_ORDER}
    dist["none"] = 0
    for d in first_days:
        dist["none" if d is None else d] += 1
    dist["_n"] = len(first_days)
    return dist


def _rate(count, n):
    return (count / n) if n else 0.0


def mention_rate(dist):
    """Fraction of samples that mention ANY weekday."""
    n = dist["_n"]
    return _rate(n - dist["none"], n)


# --------------------------------------------------------------------------- #
# 3. Seeding (PURE): deterministic per (arm, template, day, condition, dose).
# --------------------------------------------------------------------------- #
def seed_for(*keys):
    """Stable 31-bit seed from the string join of ``keys`` (blake2b). Same keys
    -> same seed (reproducible sampling); different keys -> different stream."""
    s = "|".join(str(k) for k in keys)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31 - 1)


# --------------------------------------------------------------------------- #
# 4. Readout math (PURE): the 7-day first-token logit vector (calendar-ordered).
# --------------------------------------------------------------------------- #
def _softmax7(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def logit_readout(day_logits, Y=None):
    """day_logits: (7,) calendar-ordered logits at the first generation position.
    Returns argmax_day, softmax p over the 7 days, and (if Y given) p_Y +
    whether argmax==Y. The 7-day softmax sums to 1, so mean_Y p_Y == 1/7 exactly
    under any flat/unaffected readout — that 1/7 is the wiring-check reference."""
    dl = np.asarray(day_logits, np.float64)
    p = _softmax7(dl)
    am = int(dl.argmax())
    out = {"argmax_day": CALENDAR_ORDER[am],
           "day_logits": [round(float(v), 4) for v in dl],
           "p": [round(float(v), 5) for v in p]}
    if Y is not None:
        yi = CALENDAR_ORDER.index(Y)
        out["p_Y"] = float(p[yi])
        out["argmax_eq_Y"] = (am == yi)
    return out


# --------------------------------------------------------------------------- #
# 5. Delta-vs-none aggregation (PURE). Every injection effect is reported
#    relative to the SAME template's natural propensity (the adequate-baseline
#    fix): natural propensity varies by template, so a global baseline would be
#    wrong. Each template contributes equal N, so pooling combos across templates
#    and subtracting pooled none == the mean of per-template deltas.
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def combo_effect(sample_days, Y, none_dist):
    """Per-combo generation effect vs a template's ``none_dist``.
      p_first_eq_Y      : fraction of samples whose first-mentioned day == Y.
      d_p_first_eq_Y    : that minus none's rate of naturally saying Y.
      day_mention_rate  : fraction mentioning any day.
      d_mention         : that minus none's mention rate.
    """
    D = day_distribution(sample_days)
    n = D["_n"]
    n_none = none_dist["_n"]
    p_Y = _rate(D[Y], n)
    mr = mention_rate(D)
    none_pY = _rate(none_dist[Y], n_none)
    none_mr = mention_rate(none_dist)
    return {"p_first_eq_Y": p_Y, "d_p_first_eq_Y": p_Y - none_pY,
            "day_mention_rate": mr, "d_mention": mr - none_mr, "dist": D}


# --------------------------------------------------------------------------- #
# 6. Summary schema (PURE): natural-propensity tables FIRST, then injection
#    effects as deltas, then the wiring-verification block.
# --------------------------------------------------------------------------- #
def _pooled_dist(dists):
    """Sum a list of day_distribution dicts into one pooled distribution."""
    out = {d: 0 for d in CALENDAR_ORDER}
    out["none"] = 0
    out["_n"] = 0
    for D in dists:
        for k in list(out):
            out[k] += D.get(k, 0)
    return out


def _dist_rates(dist):
    """day_distribution -> {day: rate} over the 7 days + 'none', plus mention_rate."""
    n = dist["_n"]
    r = {d: round(_rate(dist[d], n), 5) for d in CALENDAR_ORDER}
    r["none"] = round(_rate(dist["none"], n), 5)
    r["mention_rate"] = round(mention_rate(dist), 5)
    r["n"] = n
    return r


def summarize_arm(arm_result):
    """Reduce one arm's per-template records to natural-propensity tables +
    (sited arms) injection-effect deltas + the wiring-verification verdict."""
    templates = arm_result["templates"]
    has_site = arm_result["has_site"]

    # --- natural propensity (the `none` condition), the reference for all -----
    none_dists, none_logit_argmax, by_template = [], [], {}
    none_by_tid = {}
    for t in templates:
        D = day_distribution(t["none"]["sample_days"])
        none_by_tid[t["id"]] = D
        none_dists.append(D)
        none_logit_argmax.append(logit_readout(t["none"]["day_logits"])["argmax_day"])
        by_template[t["id"]] = {"family": t["family"], "greedy": t["none"]["greedy_text"],
                                "greedy_day": t["none"]["greedy_day"],
                                "rates": _dist_rates(D),
                                "logit_argmax": none_logit_argmax[-1]}
    pooled_none = _pooled_dist(none_dists)
    logit_base = {d: round(_rate(sum(1 for a in none_logit_argmax if a == d),
                                 len(none_logit_argmax)), 5) for d in CALENDAR_ORDER}
    natural = {
        "n_samples_total": pooled_none["_n"],
        "day_mention_rate": round(mention_rate(pooled_none), 5),
        "per_day_base_rate": {d: round(_rate(pooled_none[d], pooled_none["_n"]), 5)
                              for d in CALENDAR_ORDER},
        "day_distribution": {**{d: pooled_none[d] for d in CALENDAR_ORDER},
                             "none": pooled_none["none"], "_n": pooled_none["_n"]},
        "logit_argmax_base_rate": logit_base,
        "by_template": by_template,
    }
    out = {"arm": arm_result["arm"], "kind": arm_result["kind"],
           "has_site": has_site, "natural_propensity": natural}
    if not has_site:
        return out

    # --- injection effects, bucketed by (condition/pattern/@dose) -------------
    buckets = {}    # key -> list of per-combo stat dicts
    for t in templates:
        none_dist = none_by_tid[t["id"]]
        for c in t["combos"]:
            key = f"{c['condition']}/{c['pattern']}/@{c['dose']:g}"
            eff = combo_effect(c["sample_days"], c["Y"], none_dist)
            lr = logit_readout(c["day_logits"], c["Y"])
            buckets.setdefault(key, []).append({
                "Y": c["Y"], "p_first_eq_Y": eff["p_first_eq_Y"],
                "d_p_first_eq_Y": eff["d_p_first_eq_Y"],
                "day_mention_rate": eff["day_mention_rate"],
                "d_mention": eff["d_mention"], "dist": eff["dist"],
                "logit_p_Y": lr["p_Y"], "logit_argmax_eq_Y": lr["argmax_eq_Y"]})

    injection = {}
    for key, rows in sorted(buckets.items()):
        by_Y = {}
        for d in CALENDAR_ORDER:
            yr = [r for r in rows if r["Y"] == d]
            if yr:
                by_Y[d] = {"n": len(yr),
                           "mean_p_first_eq_Y": _mean([r["p_first_eq_Y"] for r in yr]),
                           "mean_d_p_first_eq_Y": _mean([r["d_p_first_eq_Y"] for r in yr]),
                           "mean_day_mention_rate": _mean([r["day_mention_rate"] for r in yr]),
                           "mean_logit_p_Y": _mean([r["logit_p_Y"] for r in yr]),
                           "logit_argmax_eq_Y_rate": _mean([1.0 if r["logit_argmax_eq_Y"] else 0.0
                                                            for r in yr])}
        injection[key] = {
            "n_combos": len(rows),
            "mean_p_first_eq_Y": _mean([r["p_first_eq_Y"] for r in rows]),
            "mean_d_p_first_eq_Y_vs_none": _mean([r["d_p_first_eq_Y"] for r in rows]),
            "mean_day_mention_rate": _mean([r["day_mention_rate"] for r in rows]),
            "mean_d_mention_vs_none": _mean([r["d_mention"] for r in rows]),
            "pooled_first_mention_dist": _pooled_dist([r["dist"] for r in rows]),
            "mean_logit_p_Y": _mean([r["logit_p_Y"] for r in rows]),
            "logit_argmax_eq_Y_rate": _mean([1.0 if r["logit_argmax_eq_Y"] else 0.0
                                             for r in rows]),
            "by_Y": by_Y,
        }
    out["injection"] = injection
    out["verification"] = _verification(injection)
    return out


def _verification(injection):
    """Wiring check on the strongest condition (inject_all/onehot/@4): does
    GENERATION move (mean d_p_first_eq_Y > eps) while the first-token LOGIT
    readout stays flat (mean_logit_p_Y ~ 1/7)? If so, the old implant readout
    that used exactly this logit mechanism was mis-wired — the injection never
    reached the readout position. Report both signals + a verdict."""
    key = f"{STRONGEST[0]}/{STRONGEST[1]}/@{STRONGEST[2]:g}"
    b = injection.get(key)
    if b is None:
        return {"note": f"strongest bucket {key} absent (subset run)"}
    chance = 1.0 / 7.0
    gen_move = (b["mean_d_p_first_eq_Y_vs_none"] or 0.0)
    logit_move = (b["mean_logit_p_Y"] or chance) - chance
    gen_moved = gen_move > MOVE_EPS
    logit_moved = logit_move > MOVE_EPS
    if gen_moved and not logit_moved:
        verdict = ("GENERATION MOVES but the first-token logit readout is flat "
                   "(~1/7): the OLD implant/first-token-logit readout was mis-wired "
                   "for this injection geometry — it did not see the effect generation shows.")
    elif gen_moved and logit_moved:
        verdict = ("both generation and the logit readout move: the old first-token "
                   "readout was correctly wired; implant genuinely biases the answer.")
    elif not gen_moved and not logit_moved:
        verdict = ("neither moves: at this arm/dose the injection does not steer the "
                   "day in free generation (consistent with the old implant near-null).")
    else:
        verdict = ("logit readout moves but generation does not: the bias is present "
                   "at the answer position yet washes out over free continuation.")
    return {"strongest_condition": key, "chance_1_over_7": round(chance, 5),
            "gen_mean_d_p_first_eq_Y": round(gen_move, 5),
            "logit_mean_p_Y": round(b["mean_logit_p_Y"] or chance, 5),
            "logit_lift_over_chance": round(logit_move, 5),
            "generation_moves": gen_moved, "logit_readout_moves": logit_moved,
            "verdict": verdict}


def build_summary(arm_summaries, meta):
    """Assemble results/opengen_summary.json: natural-propensity tables FIRST
    (per-day base rates prominent — the user's critique that a flip is only
    interpretable against how often the day is said naturally), then injection
    effects as deltas, then per-arm wiring verdicts."""
    natural = {a: s["natural_propensity"] for a, s in arm_summaries.items()}
    injection = {a: s.get("injection") for a, s in arm_summaries.items() if s["has_site"]}
    verification = {a: s.get("verification") for a, s in arm_summaries.items() if s["has_site"]}
    return {"natural_propensity_FIRST": natural,
            "injection_effects_vs_none": injection,
            "wiring_verification": verification,
            "meta": meta}


# =========================================================================== #
# RUNTIME (torch + harness + rustbpe; pod only). Everything above is pure.
# =========================================================================== #
def make_forward_fn(model, harness):
    """(ids[B,T], acts[B,T,7]|None, gate_scale) -> last-position logits [B,V]
    on device. SAME code path as harness.forward_metrics (its _scaled_gates +
    _acts_dict + model(acts=...)) minus the CE + full-logits CPU copy — a
    generation loop only needs the last row, and copying [B,T,V] every step would
    dominate the budget."""
    import torch
    sites = getattr(model, "injection_sites", None)

    def fn(ids, acts_batch, gate_scale):
        acts_dict = None
        if acts_batch is not None and sites is not None:
            at = torch.as_tensor(np.asarray(acts_batch, np.float32), device=ids.device)
            acts_dict = harness._acts_dict(model, at)
        with harness._scaled_gates(model, gate_scale):
            with torch.inference_mode():
                logits = model(ids, acts=acts_dict)
        return logits[:, -1, :].float()
    return fn


def _sample_next(logits_last, temperature, top_k, gen, n_greedy=1):
    """(B,V) logits -> (B,1) next ids. Rows [0:n_greedy] are greedy (argmax);
    the rest sample at ``temperature`` with ``top_k`` truncation, using the
    seeded generator ``gen`` (batched multinomial: one draw per row)."""
    import torch
    B, V = logits_last.shape
    out = torch.empty((B, 1), dtype=torch.long, device=logits_last.device)
    out[:n_greedy] = torch.argmax(logits_last[:n_greedy], dim=-1, keepdim=True)
    if B > n_greedy:
        lg = logits_last[n_greedy:].clone()
        if top_k and top_k > 0:
            v, _ = torch.topk(lg, min(int(top_k), V))
            lg[lg < v[:, [-1]]] = float("-inf")
        if temperature and temperature > 0:
            probs = torch.softmax(lg / float(temperature), dim=-1)
            out[n_greedy:] = torch.multinomial(probs, num_samples=1, generator=gen)
        else:
            out[n_greedy:] = torch.argmax(lg, dim=-1, keepdim=True)
    return out


def generate_completions(forward_fn, decode_fn, ids0, Lb, condition, pattern,
                         gate_scale, day_first_ids, *, n_samples, max_new,
                         temperature, top_k, seed, device="cpu", keep_texts=False):
    """Hand-rolled incremental generation (full recompute per step; no KV cache —
    correctness over speed at d12). Row 0 is greedy, rows 1..n_samples are seeded
    samples, all in ONE batched forward per step. Acts are rebuilt each step from
    ``acts_for_step`` (position-only, so identical across the batch). Returns the
    greedy completion, the sampled first-days, and the step-0 7-day logit vector.
    """
    import torch
    B = 1 + n_samples
    ids = torch.tensor([list(ids0)] * B, dtype=torch.long, device=device)
    gen = None
    if n_samples > 0 and temperature and temperature > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    dfi = torch.as_tensor(np.asarray(day_first_ids), dtype=torch.long, device=device)
    day_logits0 = None
    for step in range(max_new):
        T = ids.shape[1]
        acts_np = acts_for_step(condition, Lb, T, pattern)
        acts_batch = None if acts_np is None else np.broadcast_to(
            acts_np, (B, T, acts_np.shape[1])).copy()
        logits_last = forward_fn(ids, acts_batch, gate_scale)         # (B,V) device
        if step == 0:
            day_logits0 = logits_last[0, dfi].detach().float().cpu().numpy()
        nxt = _sample_next(logits_last, temperature, top_k, gen, n_greedy=1)
        ids = torch.cat([ids, nxt], dim=1)
    gen_ids = ids[:, 1 + Lb:].detach().cpu().tolist()
    greedy_text = decode_fn(gen_ids[0])
    sample_texts = [decode_fn(s) for s in gen_ids[1:]]
    res = {"greedy_text": greedy_text,
           "greedy_day": first_day_mentioned(greedy_text)[0],
           "sample_days": [first_day_mentioned(t)[0] for t in sample_texts],
           "day_logits": [round(float(v), 4) for v in day_logits0]}
    if keep_texts:
        res["sample_texts"] = sample_texts
    return res


def resolve_arms(spec):
    """(label, base_arm, kind). 'all' -> 3 real + 3 controls + plain baseline.
    Real arms come FIRST so their checkpoint directions warm the cache before the
    baseline_<arm> controls need them (mirrors causal.resolve_arms)."""
    real = ["trainable", "sphere", "orthogonal"]
    cfgs = ([(a, a, "real") for a in real]
            + [(f"baseline_{a}", a, "control") for a in real]
            + [("baseline", None, "plain")])
    if spec == "all":
        return cfgs
    want = [s.strip() for s in spec.split(",")]
    return [c for c in cfgs if c[0] in want]


def _combo_grid(empirical_present):
    """(condition, pattern, dose) tuples for the injection sweep (Y added later)."""
    grid = []
    for cond in CONDITIONS:
        for k in ONEHOT_DOSES:
            grid.append((cond, "onehot", k))
        if empirical_present:
            for k in EMP_DOSES:
                grid.append((cond, "empirical", k))
    return grid


def _pattern_for(kind, Y, empirical):
    return causal.onehot_pattern(Y) if kind == "onehot" else empirical[Y]


def run(args):
    import torch
    import harness
    from nanochat.tokenizer import get_tokenizer

    assert list(harness.WEEKDAY_CONCEPTS) == STORE_ORDER, \
        f"harness.WEEKDAY_CONCEPTS {harness.WEEKDAY_CONCEPTS} != STORE_ORDER {STORE_ORDER}"

    tok = get_tokenizer()
    enc = tok.enc
    bos = tok.get_bos_token_id()
    harness.set_nano_tokenizer(enc)
    day_first_ids, _ = causal._day_token_ids(enc)     # calendar-ordered ' <Day>' first tokens

    empirical = causal._load_empirical(args.empirical_json)
    items = oi.generate_items()
    if args.limit:
        items = items[:args.limit]
    grid = _combo_grid(empirical is not None)
    print(f"[opengen] {len(items)} prompts; {len(grid)} (cond/pattern/dose) x 7 days "
          f"= {len(grid) * 7} injection combos/template; empirical="
          f"{'yes' if empirical else 'no'}")

    # pre-encode prompts (BOS-prefixed; Lb = body-token count => final-prompt idx)
    prepared = []
    for it in items:
        body = enc.encode_ordinary(it["prompt"])
        assert len(body) >= 1, f"empty prompt encoding: {it['id']}"
        prepared.append(dict(it, ids0=[bos] + body, Lb=len(body)))

    gcfg = dict(n_samples=args.samples, max_new=args.max_new,
                temperature=args.temperature, top_k=args.top_k,
                device=args.device, keep_texts=args.dump_samples)

    dir_cache = {}
    arm_summaries, n_gen = {}, 0
    for label, base_arm, kind in resolve_arms(args.arms):
        if kind == "real":
            model, _ = harness.load_model(base_arm, args.device)
            dir_cache[base_arm] = causal._extract_direction(model)
        elif kind == "control":
            model, _ = harness.load_model("baseline", args.device)
            harness.attach_site(model, causal._direction_for(base_arm, harness,
                                                             args.device, dir_cache),
                                harness.GATE)
        else:                                          # plain baseline: no site
            model, _ = harness.load_model("baseline", args.device)
        has_site = getattr(model, "injection_sites", None) is not None
        fwd = make_forward_fn(model, harness)

        templates, ng = _eval_arm(label, prepared, fwd, tok.decode, day_first_ids,
                                  grid, empirical, has_site, gcfg)
        n_gen += ng
        arm_result = {"arm": label, "base_arm": base_arm, "kind": kind,
                      "has_site": has_site, "n_samples": args.samples,
                      "max_new": args.max_new, "temperature": args.temperature,
                      "top_k": args.top_k, "templates": templates}
        summ = summarize_arm(arm_result)
        arm_summaries[label] = summ
        _write_arm(label, arm_result, summ)
        print(f"[opengen] arm {label}: {ng:,} generations -> results/opengen_{label}.json")
        del model

    meta = {"n_prompts": len(items), "conditions": CONDITIONS,
            "onehot_doses": ONEHOT_DOSES, "emp_doses": EMP_DOSES,
            "n_samples": args.samples, "max_new": args.max_new,
            "temperature": args.temperature, "top_k": args.top_k,
            "gate": GATE_TRAINED, "onehot_z": ONEHOT_Z,
            "empirical_present": empirical is not None,
            "n_generations": n_gen, "forward_passes": n_gen * args.max_new,
            "store_order": STORE_ORDER, "calendar_order": CALENDAR_ORDER}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "opengen_summary.json"), "w") as f:
        json.dump(build_summary(arm_summaries, meta), f, indent=1)
    print(f"[opengen] {n_gen:,} generations ({n_gen * args.max_new:,} forwards) "
          f"-> results/opengen_summary.json")


def _eval_arm(label, prepared, fwd, decode_fn, day_first_ids, grid, empirical,
              has_site, gcfg):
    templates, n_gen = [], 0
    for it in prepared:
        rec = {"id": it["id"], "family": it["family"], "template": it["template"],
               "item": it["item"], "prompt": it["prompt"]}
        # `none` — natural propensity (Y-independent); the reference for this template
        rec["none"] = generate_completions(
            fwd, decode_fn, it["ids0"], it["Lb"], "none", None, 1.0, day_first_ids,
            seed=seed_for(label, it["id"], "none"), **gcfg)
        n_gen += 1
        if has_site:
            combos = []
            for cond, pattern_kind, dose in grid:
                for Y in CALENDAR_ORDER:
                    P = _pattern_for(pattern_kind, Y, empirical)
                    g = generate_completions(
                        fwd, decode_fn, it["ids0"], it["Lb"], cond, P, dose,
                        day_first_ids,
                        seed=seed_for(label, it["id"], Y, cond, pattern_kind, dose),
                        **gcfg)
                    combos.append(dict(condition=cond, pattern=pattern_kind,
                                       dose=dose, Y=Y, **g))
                    n_gen += 1
            rec["combos"] = combos
        templates.append(rec)
    return templates, n_gen


def _write_arm(label, arm_result, summ):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"opengen_{label}.json"), "w") as f:
        json.dump({"arm": label, "base_arm": arm_result["base_arm"],
                   "kind": arm_result["kind"], "has_site": arm_result["has_site"],
                   "config": {k: arm_result[k] for k in
                              ("n_samples", "max_new", "temperature", "top_k")},
                   "summary": summ, "templates": arm_result["templates"]}, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description="weekday-geometry open-generation battery")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="all",
                    help="'all' (3 real + 3 baseline controls + plain baseline) or a comma list")
    ap.add_argument("--samples", type=int, default=16, help="sampled completions per combo")
    ap.add_argument("--max-new", type=int, default=12, help="new tokens per completion")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--empirical-json",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "empirical_patterns.json"))
    ap.add_argument("--limit", type=int, default=0, help="cap prompt count (smoke)")
    ap.add_argument("--dump-samples", action="store_true",
                    help="store every sampled completion string (large JSON)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
