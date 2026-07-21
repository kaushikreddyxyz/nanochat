"""On-vs-off eval orchestrator, shared by every concept family: per arm x injection
{on=1.0, off=0.0} runs (1) completion accuracy/CE/margin on the family's eval set,
(2) held-out val-bpb bucketed overall/injected/after/rest, (3) CORE via an
activations-on adapter (no core_eval source diff). All forwards go through harness.py.
Writes {arm}_{on|off}.json + summary.json to --out-dir.
Run: see runs/<family>/eval/run_all.sh
"""
import argparse
import json
import math
import os
import sys

# Ensure the nanochat repo root is importable regardless of how this file is
# launched (`python -m runs.lib.eval.run_evals` OR `python runs/lib/eval/run_evals.py`),
# since we import runs.* / scripts.* / nanochat.*. Cheap, idempotent, import-safe
# (only prepends a path; no heavy imports here).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

LOUDNESS_SCALES = {"on": 1.0, "off": 0.0}
THRESHOLD_DEFAULT = 2.0

# The four arms, as (loudness_scale, ablation scope). on/off dial the site's calibrated
# channel_scale; the two ablation arms project the direction row space out of the
# residual stream (runs/lib/eval/ablation.py) and keep the oracle EXPLICITLY OFF, so
# they ask whether the model still routes concept behaviour through that subspace with
# no injected signal in play. Scope 'L0' resolves to the site's own after_block.
CONDITIONS = {
    "on":         (1.0, "none"),
    "off":        (0.0, "none"),
    "ablate_L0":  (0.0, "L0"),
    "ablate_all": (0.0, "all"),
}
ARM_NAMES = list(CONDITIONS)


def loudness_of(g):
    return CONDITIONS[g][0]


def scope_of(g):
    return CONDITIONS[g][1]

# climbmix-scored HF layout (from runs/weekdays/exp2_config.json prefetch block):
# 25 shards per repo, sid -> repos[sid // 25].
SCORED_REPOS = [
    "kaushikreddyxyz/climbmix-scored",
    "kaushikreddyxyz/climbmix-scored-overflow",
    "kaushikreddyxyz/climbmix-scored-overflow-2",
    "kaushikreddyxyz/climbmix-scored-overflow-3",
    "kaushikreddyxyz/climbmix-scored-overflow-4",
    "kaushikreddyxyz/climbmix-scored-overflow-5",
    "kaushikreddyxyz/climbmix-scored-overflow-6",
    "kaushikreddyxyz/climbmix-scored-overflow-7",
]
SCORED_PER_REPO = 25
# Held-out default: training downloaded 45 shards but only reached ~parquet 28 at
# the 1.321B-token horizon (training logs end at `pq: 28`; val split = shard 6542,
# unscored). Scored range is 0-184, so shards >= 45 are unseen AND scored; 100/101
# are comfortably beyond both bounds.
HELDOUT_SHARDS_DEFAULT = [100, 101]
CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"


def scored_repo_for_shard(sid):
    idx = sid // SCORED_PER_REPO
    if idx >= len(SCORED_REPOS):
        raise ValueError(f"shard {sid} is beyond the scored range (max {len(SCORED_REPOS)*SCORED_PER_REPO-1})")
    return SCORED_REPOS[idx]


# =========================================================================== #
# Pure reductions (no torch, no network) — unit-tested by runs/tests/.
# =========================================================================== #
def common_prefix_len(seqs):
    """Length of the longest common prefix shared by all token sequences."""
    if not seqs:
        return 0
    n = min(len(s) for s in seqs)
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return n


def option_mean_ce(per_token_ce, start, end):
    """Length-normalized CE over an option's continuation tokens.

    Harness convention: ``per_token_ce[t]`` = CE of predicting the token AT
    position t (position 0 is NaN — no context). The continuation occupies token
    positions [start, end) (end = sequence length), so the option's CE is the mean
    of per_token_ce over exactly those positions. ``start`` >= 1 (BOS is 0)."""
    start = min(max(1, start), end - 1)
    assert 1 <= start < end <= len(per_token_ce), (start, end, len(per_token_ce))
    vals = [float(per_token_ce[t]) for t in range(start, end)]
    return sum(vals) / len(vals)


def predict_from_option_ces(option_ces):
    """(pred_index, margin) for a list of per-option mean CEs. Prediction is the
    argmin CE; margin = second-best minus best (>=0; larger = more decisive).
    Ties resolve to the lowest index (deterministic)."""
    order = sorted(range(len(option_ces)), key=lambda i: (option_ces[i], i))
    best = order[0]
    margin = (option_ces[order[1]] - option_ces[best]) if len(order) > 1 else 0.0
    return best, float(margin)


def bpb_from_nats_bytes(sum_nats, sum_bytes):
    return float("inf") if sum_bytes == 0 else sum_nats / (math.log(2) * sum_bytes)


def bucket_masks(acts, valid):
    """Partition predicted-token positions into injected / after / rest — the SAME
    definition as harness.ce_report so the bpb buckets align with its CE buckets.

    ``acts`` is [T, r] aligned to INPUT positions (row t = input token t, BOS row
    zero); position t is INJECTED iff acts[t] is non-zero. ``per_token_ce[t]``
    scores the PREDICTED token t, so we bucket by the injection status of the
    predicted token: injected = acts[t] nonzero; after = acts[t-1] nonzero and not
    itself injected; rest = the remaining valid positions. ``valid[t]`` is a
    per-predicted-token bool (byte>0 target; False at position 0). Returns dict
    name -> list[bool] of length T."""
    T = len(acts)
    inj_tok = [any(float(v) != 0.0 for v in acts[t]) for t in range(T)]
    masks = {"injected": [], "after": [], "rest": []}
    for t in range(T):
        inj = bool(valid[t]) and inj_tok[t]
        aft = bool(valid[t]) and (t >= 1 and inj_tok[t - 1]) and not inj
        masks["injected"].append(inj)
        masks["after"].append(aft)
        masks["rest"].append(bool(valid[t]) and not inj and not aft)
    return masks


def merge_summary(records):
    """records: list of dicts, each {metric, arm, injection ('on'|'off'), value, n}.
    -> nested summary  metric -> arm -> injection -> {value, n}."""
    out = {}
    for rec in records:
        out.setdefault(rec["metric"], {}).setdefault(rec["arm"], {})[rec["injection"]] = {
            "value": rec["value"], "n": rec["n"]}
    return out


# =========================================================================== #
# CLI (standalone-importable; no heavy imports at build time).
# =========================================================================== #
def build_parser():
    ap = argparse.ArgumentParser(description="Injection on-vs-off eval runner (family-parameterized)")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arms to evaluate; each is the HF checkpoint FOLDER name")
    ap.add_argument("--metrics", nargs="+", default=["completion", "valbpb", "core"],
                    choices=["completion", "valbpb", "core"], help="metrics to run")
    ap.add_argument("--injection", nargs="+", default=["on", "off"], choices=ARM_NAMES,
                    help="arms: on (loudness 1.0), off (0.0), ablate_L0 / ablate_all "
                         "(0.0 + direction-subspace projection at the site's block / "
                         "every block)")
    ap.add_argument("--evalset", required=True, help="completion eval set JSONL")
    ap.add_argument("--family", required=True,
                    help="concept family in runs/lib/concepts.py: supplies the probe concepts in "
                         "STORE order, the default site name, and the metric-name prefix")
    ap.add_argument("--hf-repo", required=True, help="checkpoint HF repo for every arm")
    ap.add_argument("--step", type=int, required=True, help="checkpoint step")
    ap.add_argument("--out-dir", required=True, help="the family's results dir")
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    # val_bpb
    ap.add_argument("--heldout-shards", nargs="+", type=int, default=HELDOUT_SHARDS_DEFAULT,
                    help="scored climbmix shards for val-bpb (must be beyond the training window)")
    ap.add_argument("--valbpb-source", default="prescored", choices=["prescored", "gemma"],
                    help="prescored=positional ConceptProbeScoreSource over climbmix-scored (faithful, "
                         "matches training injection exactly); gemma=score held-out text live "
                         "(lighter: reuses the loaded gemma, no score-store download)")
    ap.add_argument("--valbpb-max-docs", type=int, default=2000, help="cap docs per held-out shard")
    ap.add_argument("--valbpb-max-tokens", type=int, default=2048,
                    help="cap tokens per doc INCLUDING BOS (train context = 2048; the rotary "
                         "cache tops out at 10x that, so uncapped megadocs would crash). Acts "
                         "are looked up on the FULL doc then truncated with the tokens.")
    ap.add_argument("--climbmix-cache", default="", help="dir for downloaded held-out climbmix parquet "
                    "(default: <out-dir>/climbmix_heldout)")
    ap.add_argument("--gemma-model", default="google/gemma-2-2b", help="gemma tokenizer for offsets")
    ap.add_argument("--layer", type=int, default=8, help="gemma layer for the probes")
    # core
    ap.add_argument("--core-max-per-task", type=int, default=500, help="examples per CORE task")
    ap.add_argument("--core-skip-gemma-when-off", action="store_true",
                    help="OPTIMIZATION: for the off CORE run, skip gemma and inject nothing "
                         "(numerically identical to loudness 0; breaks the strict same-code-path "
                         "only for speed)")
    ap.add_argument("--site-name", default=None,
                    help="injection site name in the checkpoints (default: the family name)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit-items", type=int, default=-1, help="debug: cap completion items (-1 = all)")
    # Site-less (baseline) ablation: build the projected-out subspace from EXTERNAL DoM
    # probe directions instead of an injection site. The concept lives along the raw
    # residual write direction W_dom(.)nat_std; L0 scope -> --baseline-ablate-layer, all
    # scope -> every block (same basis, matching the injected arms' ablate_all).
    ap.add_argument("--baseline-ablate-npz", default=None,
                    help="DoM probe npz (W_dom/nat_std/concepts) for a site-less model's ablation")
    ap.add_argument("--baseline-ablate-layer", type=int, default=None,
                    help="block index for the L0-scope (most-salient-layer) ablation")
    ap.add_argument("--baseline-ablate-concepts", nargs="+", default=None,
                    help="concept names to ablate (subset of the npz); default all in the npz")
    return ap


# =========================================================================== #
# Harness-dependent runners (lazy torch / harness / network imports).
# =========================================================================== #
def _harness():
    """Import the pinned harness. Kept lazy so pure helpers/CLI import torch-free.

    Imported as the top-level module ``harness`` with this file's dir on sys.path
    (NOT ``runs.lib.eval.harness``): an installed PyPI ``runs`` package
    shadows the local ``runs/`` namespace dir (a real incident — see the
    exp2_config.json comment on the file-path 'class' form). The name-based
    import is cached in sys.modules, so every caller (and test_harness.py /
    causal.py, which import it the same way) shares ONE module instance —
    set_nano_tokenizer state stays global."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import harness  # noqa: WPS433 (intentional lazy import)
    return harness


def _ablation():
    """Import the ablation module the same lazy, name-based way as the harness."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import ablation  # noqa: WPS433 (intentional lazy import)
    return ablation


def _ext_basis(ext):
    """Orthonormal raw-residual basis from external DoM probe directions (site-less
    models). The concept's raw write direction is W_dom (.) nat_std (W_dom is in
    standardized space); direction_basis SVD-orthonormalizes the selected rows."""
    import numpy as np
    ABL = _ablation()
    z = np.load(ext["npz"], allow_pickle=True)
    names = [str(c) for c in z["concepts"]]
    W = np.asarray(z["W_dom"], np.float32)
    nat_std = np.asarray(z["nat_std"], np.float32)
    pick = ext.get("concepts") or names
    rows = [names.index(c) for c in pick]
    D_raw = W[rows] * nat_std[None, :]
    return ABL.direction_basis(D_raw), rows, pick


def build_ablation_plan(model, site_name, arms, ext=None):
    """{basis, blocks: {arm: [block indices]}, info} for one loaded arm.

    Built ONCE per checkpoint: the basis comes from that model's own site, so the
    sphere and trainable arms are each ablated along their own learned directions.
    A site-less model (baseline) with ``ext`` set instead builds the basis from
    external DoM directions (--baseline-ablate-*). Non-ablation arms map to an empty
    block list, which `ablated()` treats as a no-op that registers no hooks at all."""
    ABL = _ablation()
    scopes = {scope_of(g) for g in arms}
    basis, _, info = ABL.arm_plan(model, None, site_name, "none")
    ext_layer = None
    if basis is None and ext is not None:
        basis, rows, pick = _ext_basis(ext)
        ext_layer = int(ext["layer"])
        info = {"scope": "ext-dom", "rank": int(basis.shape[0]), "r": len(rows),
                "ext_npz": os.path.basename(ext["npz"]), "salient_layer": ext_layer,
                "ext_concepts": list(pick)}
    n_layer = len(model.transformer.h)
    blocks = {}
    for g in arms:
        scope = scope_of(g)
        if basis is None or scope == "none":
            blocks[g] = []
        elif ext_layer is not None:
            blocks[g] = [ext_layer] if scope == "L0" else list(range(n_layer))
        else:
            site = model.injection_sites[site_name]
            blocks[g] = ABL.resolve_blocks(model, scope, int(site.cfg.after_block))
    info["scopes_requested"] = sorted(scopes)
    info["blocks_by_arm"] = blocks
    return {"basis": basis, "blocks": blocks, "info": info}


def _ablate_ctx(plan, model, g):
    """Context manager projecting the subspace out for arm ``g`` (no-op if not an
    ablation arm). Always used as a `with`, so hooks cannot leak between arms."""
    ABL = _ablation()
    if plan is None:
        return ABL.ablated(model, None, [])
    return ABL.ablated(model, plan["basis"], plan["blocks"].get(g, []))


def _ce1d(fm_result):
    """The single-sequence per-token CE (1D [T]) from a forward_metrics result.
    Harness returns per_token_ce as [B,T] (position 0 NaN); we forward one
    sequence at a time so B==1."""
    import numpy as np
    ptc = np.asarray(fm_result["per_token_ce"], dtype=np.float64)
    return ptc[0] if ptc.ndim == 2 else ptc


# gemma z-scores depend only on the text (never the arm/injection), so cache them for
# the whole process: the completion eval re-scores the same option strings for every
# arm, and CORE re-sees identical row texts across arms. Bounded by the number of
# unique eval texts (~KBs each).
_GEMMA_CACHE = {}


def _gemma_z_and_offsets(scorer, text):
    """(gemma_z [n_gemma, 7], gemma char-offsets [n_gemma, 2]) for one text, from
    the SAME BOS-free tokenization (so build_acts can align without re-running
    gemma). Uses GemmaScorer.score_with_offsets; falls back to score + offsets.
    Cached by text across arms/injections (z is arm-independent)."""
    hit = _GEMMA_CACHE.get(text)
    if hit is not None:
        return hit
    if hasattr(scorer, "score_with_offsets"):
        z, offsets = scorer.score_with_offsets([text])[0]
    else:
        z = scorer.score([text])[0]
        if hasattr(scorer, "gemma_offsets"):
            offsets = scorer.gemma_offsets([text])[0]
        else:
            _ids, offsets = scorer.gemma_encode(text)
    _GEMMA_CACHE[text] = (z, offsets)
    return z, offsets


def _nano_tokenizer(meta):
    """Nano tokenizer (rustbpe on the pod). load_model's meta carries no tokenizer
    by design, so build it here and register it with the harness so build_acts can
    reconstruct nano char-offsets."""
    tok = meta.get("tokenizer") if isinstance(meta, dict) else None
    if tok is None:
        from nanochat.tokenizer import get_tokenizer
        tok = get_tokenizer()
    _harness().set_nano_tokenizer(tok.enc)
    return tok


def _acts_with_bos(acts_body, r):
    """Prepend a zero row for the BOS token so acts align 1:1 with [bos]+ids."""
    import numpy as np
    if acts_body is None:
        acts_body = np.zeros((0, r), dtype=np.float32)
    acts_body = np.asarray(acts_body, dtype=np.float32).reshape(-1, r)
    return np.concatenate([np.zeros((1, r), np.float32), acts_body], axis=0)


# --------------------------------------------------------------------------- #
# (1) the family's completion eval set
# --------------------------------------------------------------------------- #
def run_completion(model, meta, scorer, items, device, injection, site_name, threshold, r,
                   limit=-1, abl_plan=None):
    """Score the completion eval set at each requested injection setting. Acts are
    computed ONCE per option (gemma is expensive) and forwarded at every loudness scale,
    so on/off see byte-identical activations. Returns
    {injection: {overall/per-category stats}}."""
    harness = _harness()
    tok = _nano_tokenizer(meta)
    bos = tok.get_bos_token_id()
    if limit and limit > 0:
        items = items[:limit]

    # accumulator: injection -> tier/category -> {n, correct, sum_answer_ce, sum_margin}
    acc = {g: {} for g in injection}

    # If EVERY requested arm has loudness 0 (an off-only or ablation-only invocation),
    # the site is an exact no-op and the acts can never matter — so skip the gemma pass
    # entirely and forward vanilla. harness.forward_metrics documents acts=None as
    # bit-identical to loudness 0. This is what makes an ablation-only process cheap
    # enough to run beside the `on` process instead of after it.
    need_acts = any(loudness_of(g) > 0.0 for g in injection)
    if not need_acts:
        print("[run_evals] all arms are loudness 0 -> skipping gemma for completion")

    for it in items:
        options = it["options"]
        answer_index = options.index(it["answer"])
        fulls = [it["prompt"] + " " + opt for opt in options]
        seqs = [tok.encode(f, prepend=bos) for f in fulls]  # [bos] + encode_ordinary(f)
        start = common_prefix_len(seqs)
        if start < 1:
            start = 1  # never slice before the first predicted token

        # per injection setting -> list of option mean-CEs
        opt_ce = {g: [] for g in injection}
        for i, f in enumerate(fulls):
            ids = seqs[i]
            if need_acts:
                gemma_z, offsets = _gemma_z_and_offsets(scorer, f)
                nano_body = ids[1:]
                acts_body = harness.build_acts(f, nano_body, gemma_z, offsets,
                                               threshold=threshold)
                acts = _acts_with_bos(acts_body, r)[None]  # [1,T,r] aligned to [1,T] ids
            else:
                acts = None
            for g in injection:
                with _ablate_ctx(abl_plan, model, g):
                    fm = harness.forward_metrics(model, ids, acts,
                                                 loudness_scale=loudness_of(g))
                ce = _ce1d(fm)
                opt_ce[g].append(option_mean_ce(ce, start, len(ids)))

        for g in injection:
            pred, margin = predict_from_option_ces(opt_ce[g])
            # Group by TIER when the item has one (the v2 sets): the difficulty ladder
            # is the axis the four-arm analysis compares within. Falls back to
            # `category` for v1 sets, which have no tier.
            key = it.get("tier") or it["category"]
            c = acc[g].setdefault(key, {"n": 0, "correct": 0,
                                        "sum_answer_ce": 0.0, "sum_margin": 0.0})
            c["n"] += 1
            c["correct"] += int(pred == answer_index)
            c["sum_answer_ce"] += opt_ce[g][answer_index]
            c["sum_margin"] += margin

    # finalize
    out = {}
    for g in injection:
        cats = {}
        tot = {"n": 0, "correct": 0, "sum_answer_ce": 0.0, "sum_margin": 0.0}
        for cat, c in sorted(acc[g].items()):
            cats[cat] = {"n": c["n"], "accuracy": c["correct"] / c["n"],
                         "mean_answer_ce": c["sum_answer_ce"] / c["n"],
                         "mean_margin": c["sum_margin"] / c["n"]}
            for k in tot:
                tot[k] += c[k]
        out[g] = {"overall": {"n": tot["n"], "accuracy": tot["correct"] / tot["n"],
                              "mean_answer_ce": tot["sum_answer_ce"] / tot["n"],
                              "mean_margin": tot["sum_margin"] / tot["n"]},
                  "by_category": cats}
    return out


# --------------------------------------------------------------------------- #
# (2) val_bpb on held-out scored climbmix
# --------------------------------------------------------------------------- #
def _download_climbmix_shard(sid, dest_dir):
    """Fetch one climbmix parquet shard's TEXT into dest_dir (idempotent)."""
    import requests
    os.makedirs(dest_dir, exist_ok=True)
    fname = f"shard_{sid:05d}.parquet"
    path = os.path.join(dest_dir, fname)
    if os.path.exists(path):
        return path
    url = f"{CLIMBMIX_BASE_URL}/{fname}"
    tmp = path + ".tmp"
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.rename(tmp, path)
    return path


def _iter_shard_texts(path, max_docs):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    row = 0
    for rg in range(pf.num_row_groups):
        for v in pf.read_row_group(rg, columns=["text"]).column(0):
            if max_docs > 0 and row >= max_docs:
                return
            yield row, v.as_py()
            row += 1


def _stage_scored_shard(sid, staging_dir):
    """Stage one held-out shard's climbmix-scored files into a single local dir:
    the metadata JSON (columns/quant/corpus_stats) live ONLY in the primary repo,
    while the per-shard docs/scores live in the sid's overflow repo (per_repo=25).
    ConceptProbeScoreSource reads a local dir with `os.path.isdir` fast paths, so
    co-locating all five files there avoids the split-repo 404s. ~8.7 GB/shard."""
    from huggingface_hub import hf_hub_download
    os.makedirs(staging_dir, exist_ok=True)
    for name in ("columns.json", "quant.json", "corpus_stats.json"):
        hf_hub_download(SCORED_REPOS[0], name, repo_type="dataset", local_dir=staging_dir)
    repo = scored_repo_for_shard(sid)
    for name in (f"docs_{sid:05d}.jsonl", f"scores_{sid:05d}.npy"):
        hf_hub_download(repo, name, repo_type="dataset", local_dir=staging_dir)
    return staging_dir


def _make_prescored_source(sid, nano_enc, gemma_encode, layer, threshold, staging_dir,
                           family, concepts):
    # ConceptProbeScoreSource via the harness re-export (absolute file-path import
    # of runs/lib/probe_source.py — immune to the PyPI 'runs' shadowing).
    H = _harness()
    assert H.ConceptProbeScoreSource is not None, "runs/lib/probe_source.py failed to import"
    local = _stage_scored_shard(sid, staging_dir)
    return H.ConceptProbeScoreSource(
        local, [sid], layer=layer, nano_enc=nano_enc, family=family,
        gemma_encode=gemma_encode, concepts=concepts, align_policy="max",
        noise_sigma=0.0, present_z=threshold, name=f"{family}-valbpb")


def run_valbpb(model, meta, scorer, args, device, injection, site_name, threshold, r,
               abl_plan=None):
    """bits-per-byte over held-out scored climbmix docs, bucketed injected/after/rest.
    Acts computed once per doc, forwarded at every loudness scale. Returns
    {injection: {bucket: {bpb, mean_ce, n_tokens}}}."""
    import numpy as np
    harness = _harness()
    tok = _nano_tokenizer(meta)
    bos = tok.get_bos_token_id()
    nano_enc = tok.enc
    from nanochat.tokenizer import get_token_bytes
    token_bytes = get_token_bytes(device="cpu").tolist()

    gemma_encode = None
    if args.valbpb_source == "prescored":
        from nanochat.injection.sources import _default_gemma_encode
        gemma_encode = _default_gemma_encode(args.gemma_model)

    cache_dir = args.climbmix_cache or os.path.join(args.out_dir, "climbmix_heldout")
    # injection -> bucket -> [sum_nats, sum_bytes, n_tokens]
    agg = {g: {b: [0.0, 0, 0] for b in ("overall", "injected", "after", "rest")} for g in injection}

    for sid in args.heldout_shards:
        shard_path = _download_climbmix_shard(sid, cache_dir)
        src = (_make_prescored_source(sid, nano_enc, gemma_encode, args.layer, threshold,
                                      os.path.join(cache_dir, f"scored_{sid:05d}"),
                                      args.family, harness.family_concepts(args.family))
               if args.valbpb_source == "prescored" else None)
        for row, text in _iter_shard_texts(shard_path, args.valbpb_max_docs):
            if not text:
                continue
            nano_body = nano_enc.encode_ordinary(text)
            if not nano_body:
                continue
            if src is not None:
                acts_body, _ = src.lookup_by_row(sid, row, text, len(nano_body))
                if acts_body is None:  # shard miss / tokenizer drift -> no injection
                    acts_body = np.zeros((len(nano_body), r), np.float32)
            else:
                gemma_z, offsets = _gemma_z_and_offsets(scorer, text)
                acts_body = harness.build_acts(text, nano_body, gemma_z, offsets, threshold=threshold)
            # Truncate AFTER the acts lookup (the prescored positional join needs the
            # full-doc token count to match), keeping ids/acts aligned.
            if args.valbpb_max_tokens > 0 and len(nano_body) > args.valbpb_max_tokens - 1:
                nano_body = list(nano_body)[:args.valbpb_max_tokens - 1]
                acts_body = np.asarray(acts_body, np.float32)[:args.valbpb_max_tokens - 1]
            acts = _acts_with_bos(acts_body, r)  # [T, r], acts[t] = input token t
            ids = [bos] + list(nano_body)
            T = len(ids)
            # per_token_ce[t] scores PREDICTED token ids[t] (t>=1); its bytes are
            # token_bytes[ids[t]] (0 for BOS/specials -> excluded, per evaluate_bpb).
            valid = [t >= 1 and 0 <= ids[t] < len(token_bytes) and token_bytes[ids[t]] > 0
                     for t in range(T)]
            masks = bucket_masks(acts, valid)
            for g in injection:
                with _ablate_ctx(abl_plan, model, g):
                    fm = harness.forward_metrics(model, ids, acts[None],
                                                 loudness_scale=loudness_of(g),
                                                 return_logits=False)  # CE-only: skip ~268MB/doc copy
                ce = _ce1d(fm)
                for t in range(T):
                    if not valid[t]:
                        continue
                    nats = float(ce[t])
                    nbytes = token_bytes[ids[t]]
                    for b in ("overall", "injected", "after", "rest"):
                        if b == "overall" or masks[b][t]:
                            agg[g][b][0] += nats
                            agg[g][b][1] += nbytes
                            agg[g][b][2] += 1

    out = {}
    for g in injection:
        buckets = {}
        for b, (sn, sb, ntok) in agg[g].items():
            buckets[b] = {"bpb": bpb_from_nats_bytes(sn, sb),
                          "mean_ce": (sn / ntok) if ntok else float("nan"),
                          "n_tokens": ntok, "n_bytes": sb}
        out[g] = buckets
    return out


# --------------------------------------------------------------------------- #
# (3) CORE via activations-on adapter (no core_eval source diff).
# --------------------------------------------------------------------------- #
class CoreActivationsAdapter:
    """Callable that mimics a nanochat base model for nanochat.core_eval, but injects
    the family's activations. core_eval.forward_model calls ``model(input_ids)`` ->
    logits; we intercept, compute per-row acts (decode ids -> text -> gemma ->
    build_acts) and forward them under harness.scaled_loudness(loudness_scale) — the
    SAME on/off mechanism as forward_metrics (scale 0.0 = exact no-op)."""

    def __init__(self, model, tok, scorer, harness, loudness_scale, site_name, threshold, r,
                 device, skip_gemma=False, z_cache=None, abl_plan=None, arm_name="on"):
        self.abl_plan = abl_plan
        self.arm_name = arm_name
        self.model = model
        self.tok = tok
        self.bos = tok.get_bos_token_id()
        self.scorer = scorer
        self.harness = harness
        self.loudness_scale = loudness_scale
        self.site_name = site_name
        self.threshold = threshold
        self.r = r
        self.device = device
        # baseline arm has no site -> acts are ignored; skip the gemma cost entirely.
        # When loudness_scale==0 and skip_gemma is NOT set we still build acts and let
        # the *0 zero them (strict same-code-path off run).
        self.has_site = getattr(model, "_injection_by_block", None) is not None and \
            site_name in getattr(model, "injection_sites", {})
        self.skip = skip_gemma or not self.has_site
        self.z_cache = z_cache if z_cache is not None else {}
        self.n_rows = 0      # rows seen (non-skip)
        self.n_drift = 0     # decode->re-encode mismatches (row fell back to zero acts)

    def get_device(self):
        return self.model.get_device()

    def _row_acts(self, ids_row):
        import numpy as np
        body = list(ids_row[1:])
        while body and body[-1] == self.bos:  # strip right-pad (bos is the pad token)
            body.pop()
        acts = np.zeros((len(ids_row), self.r), np.float32)
        if not body:
            return acts
        self.n_rows += 1
        text = self.tok.decode(body)
        nano_ids = self.tok.enc.encode_ordinary(text)
        if nano_ids != body:  # decode/re-encode drift -> fail safe to zero (no injection)
            self.n_drift += 1
            return acts
        cached = self.z_cache.get(text)
        if cached is None:
            gemma_z, offsets = _gemma_z_and_offsets(self.scorer, text)
            body_acts = self.harness.build_acts(text, nano_ids, gemma_z, offsets, threshold=self.threshold)
            cached = np.asarray(body_acts, np.float32).reshape(-1, self.r)
            self.z_cache[text] = cached
        acts[1:1 + cached.shape[0]] = cached
        return acts

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        import numpy as np
        import torch
        if self.skip:
            return self.model(input_ids)
        rows = input_ids.detach().cpu().tolist()
        B, T = input_ids.shape
        acts = np.zeros((B, T, self.r), np.float32)
        for b, row in enumerate(rows):
            acts[b] = self._row_acts(row)
        acts_t = torch.from_numpy(acts).to(input_ids.device)
        # on/off via the SAME knob as every other metric: scale the site's calibrated
        # channel_scale (harness.scaled_loudness; 0.0 = exact no-op), never the acts.
        # test_consolidation.py asserts loudness-0, acts*0 and the vanilla forward are
        # bit-identical.
        # Ablation (if this arm is one) wraps the forward, so CORE is measured under
        # exactly the same intervention as the narrow sets and val-bpb.
        with _ablate_ctx(self.abl_plan, self.model, self.arm_name):
            with self.harness.scaled_loudness(self.model, float(self.loudness_scale)):
                return self.model(input_ids, acts={self.site_name: acts_t})


# Built acts per CORE row text (post-threshold [T,r]); text -> acts is arm- and
# injection-independent, so share across ALL arms/injections (saves a full gemma pass
# per extra injected arm; ~1 GB RAM at 500-per-task scale).
_CORE_ACTS_CACHE = {}


def run_core(model, meta, scorer, args, device, injection, site_name, threshold, r,
             abl_plan=None):
    """CORE per arm via the adapter. Returns {arm: {core_metric, n_tasks}}."""
    harness = _harness()
    from scripts.base_eval import evaluate_core
    tok = _nano_tokenizer(meta)
    z_cache = _CORE_ACTS_CACHE  # shared across arms AND the on/off passes
    out = {}
    for g in injection:
        # Loudness 0 makes the site an exact no-op, so the acts are irrelevant and the
        # (expensive) gemma pass is pure waste. That covers `off` AND both ablation
        # arms — the single biggest time saving in the four-arm matrix.
        skip = args.core_skip_gemma_when_off and loudness_of(g) == 0.0
        adapter = CoreActivationsAdapter(model, tok, scorer, harness, loudness_of(g),
                                         site_name, threshold, r, device, skip_gemma=skip,
                                         z_cache=z_cache, abl_plan=abl_plan, arm_name=g)
        res = evaluate_core(adapter, tok, device, max_per_task=args.core_max_per_task)
        out[g] = {"core_metric": res["core_metric"], "n_tasks": len(res["centered_results"]),
                  "centered_results": res["centered_results"],
                  "adapter_rows": adapter.n_rows, "adapter_drift_rows": adapter.n_drift}
        if adapter.n_drift:
            print(f"[run_evals] WARNING: CORE adapter decode/re-encode drift on "
                  f"{adapter.n_drift}/{adapter.n_rows} rows (those rows got ZERO acts)")
    return out


# =========================================================================== #
# Orchestration
# =========================================================================== #
def _summary_records(arm, injection, completion_res, valbpb_res, core_res, family):
    """Flatten each metric's headline number into summary records. ``family`` prefixes
    the completion accuracy/CE metric names."""
    recs = []
    for g in injection:
        if completion_res is not None:
            o = completion_res[g]["overall"]
            recs.append({"metric": f"{family}_accuracy", "arm": arm, "injection": g,
                         "value": o["accuracy"], "n": o["n"]})
            recs.append({"metric": f"{family}_answer_ce", "arm": arm, "injection": g,
                         "value": o["mean_answer_ce"], "n": o["n"]})
        if valbpb_res is not None:
            for bucket in ("overall", "injected", "after", "rest"):
                bd = valbpb_res[g][bucket]
                recs.append({"metric": f"valbpb_{bucket}", "arm": arm, "injection": g,
                             "value": bd["bpb"], "n": bd["n_tokens"]})
        if core_res is not None:
            recs.append({"metric": "core_metric", "arm": arm, "injection": g,
                         "value": core_res[g]["core_metric"], "n": core_res[g]["n_tasks"]})
    return recs


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    injection = args.injection

    import torch  # noqa: WPS433
    harness = _harness()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --evalset takes a single .jsonl OR a directory of tier files (the v2 layout:
    # <family>_v2/T0_copy.jsonl, T1_assoc.jsonl, ...). A directory is read in sorted
    # order so the item sequence is reproducible.
    if os.path.isdir(args.evalset):
        paths = sorted(os.path.join(args.evalset, f)
                       for f in os.listdir(args.evalset) if f.endswith(".jsonl"))
        assert paths, f"no .jsonl files in {args.evalset}"
    else:
        paths = [args.evalset]
    items = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            items += [json.loads(line) for line in f if line.strip()]
    print(f"[run_evals] evalset: {len(items)} items from {len(paths)} file(s)")

    concepts = harness.family_concepts(args.family)
    site_name_default = args.site_name or args.family
    lm_kw = {"hf_repo": args.hf_repo, "step": args.step}

    # Gemma is only needed to score an injection-ON arm (loudness>0) or gemma val-bpb.
    # A baseline / off / ablation-only run is loudness 0 everywhere -> skip the ~5GB load.
    need_gemma = any(loudness_of(g) > 0.0 for g in injection) or \
        ("valbpb" in args.metrics and args.valbpb_source == "gemma")
    scorer = None
    if need_gemma:
        scorer = harness.GemmaScorer(device, concepts, layer=args.layer)

    ext_abl = None
    if args.baseline_ablate_npz:
        ext_abl = {"npz": args.baseline_ablate_npz, "layer": args.baseline_ablate_layer,
                   "concepts": args.baseline_ablate_concepts}

    all_records = []
    for arm in args.arms:
        print(f"[run_evals] loading arm={arm}")
        model, meta = harness.load_model(arm, device, **lm_kw)
        site_name = (meta.get("site_name") if isinstance(meta, dict) else None) or site_name_default
        threshold = (meta.get("threshold") if isinstance(meta, dict) else None) or THRESHOLD_DEFAULT
        r = len(concepts)

        abl_plan = build_ablation_plan(model, site_name, injection, ext=ext_abl)
        print(f"[run_evals] {arm}: ablation {abl_plan['info']}")

        completion_res = valbpb_res = core_res = None
        if "completion" in args.metrics:
            print(f"[run_evals] {arm}: {args.family} completion ({len(items)} items) "
                  f"injection={injection}")
            completion_res = run_completion(model, meta, scorer, items, device, injection,
                                            site_name, threshold, r, limit=args.limit_items,
                                            abl_plan=abl_plan)
        if "valbpb" in args.metrics:
            print(f"[run_evals] {arm}: val_bpb shards={args.heldout_shards} src={args.valbpb_source}")
            valbpb_res = run_valbpb(model, meta, scorer, args, device, injection, site_name,
                                    threshold, r, abl_plan=abl_plan)
        if "core" in args.metrics:
            print(f"[run_evals] {arm}: CORE (max_per_task={args.core_max_per_task}) "
                  f"injection={injection}")
            core_res = run_core(model, meta, scorer, args, device, injection, site_name,
                                threshold, r, abl_plan=abl_plan)

        detail = {"arm": arm, "injection": injection, "completion": completion_res,
                  "valbpb": valbpb_res, "core": core_res}
        for g in injection:
            with open(os.path.join(args.out_dir, f"{arm}_{g}.json"), "w", encoding="utf-8") as f:
                json.dump({k: (v[g] if isinstance(v, dict) and g in v else v)
                           for k, v in detail.items() if k != "injection"}, f, indent=2)
        all_records += _summary_records(arm, injection, completion_res, valbpb_res, core_res,
                                        family=args.family)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = merge_summary(all_records)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_evals] wrote summary.json ({len(all_records)} records) -> {args.out_dir}")


if __name__ == "__main__":
    main()
