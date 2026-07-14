"""
Train a base model WITH activation injection (nanochat.injection). Same core
flags/behavior as scripts/base_train.py — which stays stock for non-injection
runs — plus the injection surface. Exactly one of --activation-store /
--activation-config is required. From root directory of the project:

python -m scripts.injection_train -- --activation-store <dir>

or distributed as:

torchrun --nproc_per_node=8 -m scripts.injection_train -- --activation-store <dir>

Reproducing the v1 run: pass --gate 0.05 explicitly (default is 1.0).
Tiny CPU smoke: add --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20

NOTE this file is a deliberate fork of scripts/base_train.py: the shared body
is kept byte-identical, so `diff scripts/base_train.py scripts/injection_train.py`
shows only the injection hunks. Keep it that way when either file changes.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.injection.sites import (InjectionCfg, reassert_optimizability, calibrate_auto_gate,
                                       classify_gate_spec, dial_gate_from_loudness, discover_loudness_json,
                                       assert_gate_identical_across_ranks)
from nanochat.injection.sources import open_store, RuntimeProbeScoreSource, load_source_class
from nanochat.injection.activation_dataloader import acts_data_loader_buffered
from nanochat.injection.buffering import coordinate_rebuffer, duty_cycle_forecast, rebuffer_progress, shard_cover_seconds, size_prefetch_streams
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model with activation injection")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--wandb-project", type=str, default="nanochat", help="wandb project name")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save model checkpoints every N steps (-1 = only at end, 0 = never save)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Reproducibility & checkpoint controls
parser.add_argument("--seed", type=int, default=1337, help="RNG seed for weight init (same seed => identical init across runs/ranks; the pretraining dataloader is already deterministic)")
parser.add_argument("--save-optimizer", type=str, default="every", choices=["every", "final", "never"], help="when to save optimizer state with checkpoints: 'every' checkpoint (resume anywhere), 'final' step only (resume from end, lighter), or 'never' (smallest, not resumable)")
parser.add_argument("--compress-checkpoints", type=int, default=1, help="gzip-compress checkpoint .pt files (1=on, 0=off); loading auto-detects compression")
parser.add_argument("--checkpoint-compress-level", type=int, default=4, help="gzip level 1-9 for checkpoint compression (higher = smaller files, slower)")
# Model ablations
parser.add_argument("--no-value-embeds", action="store_true", help="zero and freeze the ResFormer value-embedding tables so they neither contribute nor learn (matched no-value-embeds control; identical to the default run in every other respect given --seed)")
# Injection surface (exactly one of --activation-store / --activation-config required)
parser.add_argument("--activation-store", type=str, default="", help="activation store dir (activations.int8/index.npy/meta.json[/P.npy] from scripts/precompute_activations.py): ONE tabular site named 'acts' with a frozen direction (the store's P if present, else seeded orthonormal)")
parser.add_argument("--after-block", type=int, default=7, help="0-based block index for the single-store site; activations are added right AFTER transformer.h[N]")
parser.add_argument("--gate", type=str, default="1.0", help="LOUDNESS DIAL in donor units: a plain number resolves to absolute rms(gate) = dial × L_ref, where L_ref = gemma's native median 54-concept packet loudness at the source's gemma layer (loudness.json subspace_total.ridge[L].p50; measured ~0.081 at L6/L8, 0.086 at L14). Default 1.0 = standard (donor-native) loudness; 0 = exactly off; per-channel mix donor-proportional. REQUIRES loudness.json + a source gemma layer (hard error otherwise). Absolute escapes: 'abs:<n>' raw stream fraction (old semantics; the old 0.05 default == abs:0.05 ≈ dial 0.62), 'auto[:target]' absolute channel-equalized. 'donor[:stat]' = dial alias (donor == 1.0; donor:p95 == p95/p50 ≈ 1.6).")
parser.add_argument("--gate-k", type=int, default=256, help="docs sampled (seeded) for --gate dial/auto/donor calibration")
parser.add_argument("--gate-min-docs", type=int, default=16, help="fail --gate dial/auto/donor if the source yields fewer sample docs than this")
parser.add_argument("--loudness-json", type=str, default="", help="dial/donor gates: loudness.json location (local file/dir OR HF dataset repo id). Empty = the source's own store root, else fall back to kaushikreddyxyz/climbmix-scored (logged loudly; unavailable = hard error).")
parser.add_argument("--lookup-workers", type=int, default=0, help="threads for per-doc activation lookups in the ride-along loader (0=serial); overlaps runtime gemma scoring with training")
parser.add_argument("--noise-sigma", type=float, default=0.15, help="gaussian noise std on standardized activations at load time, deterministic per doc-content hash; 0 disables")
parser.add_argument("--activation-config", type=str, default="", help="JSON for the multi-site form: {\"sites\": [InjectionCfg dicts], \"sources\": {site: {\"kind\": \"qwen-encoder\"|\"probe-scores\", \"dir\": ..., \"noise_sigma\": ..., \"align_policy\": \"mean\"|\"last\"}}}; mutually exclusive with --activation-store. See nanochat/injection/README.md")
parser.add_argument("--compact-tokens", action="store_true", help="OPT-IN: cut nanochat tokens at gemma boundaries so each nests in one gemma token (needs a probe-scores-runtime source). CHANGES the training token stream (inflates token count) — not baseline-comparable. Default OFF (standard tokenization + overlap-mean alignment).")
# Staying-ahead guarantees (Amendment 2): the activation source must keep up with training.
parser.add_argument("--starvation-threshold", type=float, default=0.15, help="warn if time-blocked-waiting-on-activations exceeds this fraction of step time over the window")
parser.add_argument("--starvation-window", type=int, default=50, help="rolling window (steps) for the blocked-fraction starvation check")
parser.add_argument("--starvation-abort", action="store_true", help="hard-fail (instead of warn) when the blocked fraction exceeds --starvation-threshold, or when the prefill duty-cycle forecast is < 1x")
parser.add_argument("--target-step-time", type=float, default=0.5, help="assumed seconds/step used for the prefill duty-cycle forecast (consumption tok/s = total_batch_size / world / this)")
# Video-player buffering / backpressure (token-denominated, per rank). Prefill warms the
# activation buffer before step 1; a dry buffer triggers a DDP-coordinated rebuffer.
parser.add_argument("--prefill-tokens", type=int, default=-1, help="warm the activation buffer to this many tokens before step 1 (-1 = 4x per-rank batch tokens)")
parser.add_argument("--rebuffer-tokens", type=int, default=-1, help="when the buffer runs dry, pause consumption until it refills to this many tokens, then resume (-1 = prefill/2)")
parser.add_argument("--dry-tokens", type=int, default=-1, help="buffer depth (tokens) below which the buffer counts as dry and a rebuffer is triggered (-1 = one per-rank batch)")
parser.add_argument("--buffer-max-tokens", type=int, default=-1, help="backpressure cap: production blocks once the buffer reaches this many tokens (-1 = 2x prefill)")
parser.add_argument("--buffer-check-every", type=int, default=1, help="steps between the DDP-coordinated buffer-low check (one cheap int all_reduce)")
parser.add_argument("--download-streams", type=int, default=-1, help="concurrent shard download streams for prefetch sources that don't set 'streams' (-1 = size from prefill bandwidth vs forecast consumption)")
parser.add_argument("--shard-bytes", type=float, default=8.7e9, help="approx bytes per score shard, for prefetch stream sizing")
parser.add_argument("--tokens-per-shard", type=float, default=-1.0, help="approx nanochat tokens covered by one score shard, for prefetch stream sizing (-1 = skip auto-sizing, honor --download-streams / configured streams)")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# Reproducibility: pin weight init with a configurable seed so paired runs (e.g. the oracle
# control vs treatment) initialize identically. This re-seed is ungated by rank, so all ranks
# init the same weights — genuinely required here because nanochat uses no DDP/param broadcast
# (DistMuonAdamW only all-reduces gradients), so divergent init would never resync. Cross-rank
# parity already held via compute_init's fixed seed(42); the value added here is making the
# seed configurable (overriding that 42). torch.manual_seed already seeds every CUDA/MPS device
# generator internally, so no explicit cuda.manual_seed_all is needed. NOTE: the seed pins only
# *weight init* — data order is pinned independently, by the deterministic loader (rank-strided,
# no shuffle/RNG) given a fixed shard set + world_size + batch + seq_len, not by this seed.
# Residual GPU-atomic nondeterminism in the backward pass means run-to-run bit-exactness is
# still not guaranteed (the init lottery is fixed, the trajectory is not).
torch.manual_seed(args.seed)
print0(f"Seeded RNGs with seed={args.seed}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=args.wandb_project, name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# -----------------------------------------------------------------------------
# Injection sites: built BEFORE any checkpoint load so injected checkpoints
# (which carry injection_sites.* keys) load cleanly. Site inits draw no RNG
# from the global stream.
def _open_injection_source(name, spec, cfg, tok, seed, nano_enc=None, gemma_encode=None):
    """Store kinds -> open_store; 'probe-scores-runtime' -> RuntimeProbeScoreSource
    (applies gold gemma scores at runtime, positional join via the ride-along
    loader). 'probe-scores-live' is programmatic (needs a score_fn), not JSON-wired.
    ``nano_enc``/``gemma_encode`` override the runtime source's tokenizers (compact
    mode passes the gemma-boundary-respecting enc + shared gemma tokenizer)."""
    kind = spec.get("kind")
    if kind == "probe-scores-runtime":
        from scripts.precompute_activations import parse_shard_range
        sh = spec["shards"]
        shards = parse_shard_range(sh) if isinstance(sh, str) else [int(s) for s in sh]
        # Optional experiment-side subclass hook: "class" names a
        # RuntimeProbeScoreSource subclass ("path/to/file.py:Class" preferred,
        # "pkg.mod:Class" also accepted — see load_source_class), and "kwargs"
        # is passed through to its constructor. No behavior change when absent.
        src_cls = RuntimeProbeScoreSource
        if spec.get("class"):
            src_cls = load_source_class(spec["class"])
            assert issubclass(src_cls, RuntimeProbeScoreSource), \
                f"source 'class' {spec['class']!r} must subclass RuntimeProbeScoreSource"
        return src_cls(
            spec["score_shards_dir_or_repo"], shards, layer=int(spec.get("layer", 8)),
            nano_enc=nano_enc or tok.enc, gemma_encode=gemma_encode, concepts=spec.get("concepts"),
            gemma_model=spec.get("gemma_model", "google/gemma-2-2b"),
            align_policy=spec.get("align_policy", "mean"),
            climbmix_dir=spec.get("climbmix_dir"), index_path=spec.get("index_path"),
            build_hash_index=bool(spec.get("build_hash_index", False)),
            noise_sigma=float(spec.get("noise_sigma", 0.15)), seed=seed, name=name,
            **dict(spec.get("kwargs") or {}))
    return open_store(spec["dir"], noise_sigma=float(spec.get("noise_sigma", 0.15)),
                      seed=seed, name=name, expect_kind=kind)

def _gate_str(g):
    if isinstance(g, (list, tuple)):
        import numpy as _np; a = _np.asarray(g, float)
        return f"vec[r={a.size}] rms={float(_np.sqrt((a**2).mean())):.4f} min={a.min():.4f} max={a.max():.4f}"
    return str(g)

assert bool(args.activation_store) != bool(args.activation_config), \
    "pass exactly one of --activation-store / --activation-config (vanilla runs use scripts/base_train.py)"
if args.activation_store:
    with open(os.path.join(args.activation_store, "meta.json")) as f:
        _store_meta = json.load(f)
    injection_cfgs = [InjectionCfg(
        name="acts", r=int(_store_meta["r"]), after_block=args.after_block,
        gate=args.gate, trainable_direction=False,
        direction_seed=int(_store_meta.get("p_seed", 1337)))]
    injection_source_specs = {"acts": {"kind": _store_meta.get("source_kind"),
                                       "dir": args.activation_store,
                                       "noise_sigma": args.noise_sigma}}
else:
    with open(args.activation_config) as f:
        _inj_spec = json.load(f)
    injection_cfgs = [InjectionCfg(**d) for d in _inj_spec["sites"]]
    injection_source_specs = dict(_inj_spec["sources"])
    assert set(injection_source_specs) == {c.name for c in injection_cfgs}, \
        "--activation-config: 'sources' keys must match the site names exactly"

base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1

# Classify gate specs (grammar: sites.classify_gate_spec). A PLAIN NUMBER is a
# loudness DIAL in donor units — resolved below to an absolute per-channel gate
# (rms(gate) = dial × L_ref from loudness.json) at startup; "abs:<n>"/"auto" are
# absolute; "donor[:stat]" is a dial alias; dial 0 is an exact off without the
# artifact. Placeholders until calibration/load. Sources open BEFORE the sites
# so calibration can sample them independently; they are loader-side data
# plumbing, never model state.
_auto_targets = {}       # name -> auto absolute target loudness
_dial_specs = {}         # name -> {"dial": float} | {"stat": str}  (donor alias)
for _cfg in injection_cfgs:
    _mode, _val = classify_gate_spec(_cfg.gate)
    if _mode == "auto":
        _auto_targets[_cfg.name] = _val
        _cfg.gate = float(_val)                 # placeholder until calibration/load
    elif _mode == "abs":
        _cfg.gate = float(_val)                 # raw stream fraction (old semantics)
    elif _mode == "donor":
        _dial_specs[_cfg.name] = {"stat": _val}
        _cfg.gate = 0.05                        # placeholder until calibration/load
    elif _mode == "dial":
        if _val == 0.0:
            _cfg.gate = 0.0                     # exact off — no loudness.json needed
        else:
            _dial_specs[_cfg.name] = {"dial": float(_val)}
            _cfg.gate = 0.05                    # placeholder until calibration/load
    else:                                       # explicit absolute per-channel vector
        _cfg.gate = [float(g) for g in _val]

# Resuming a dial/auto-gate run: the calibrated per-channel vector must be in
# the cfg BEFORE the sites are built — load_state_dict(assign=True) enforces
# shapes, so a scalar-placeholder site cannot load a vector-gate checkpoint. Read
# it from the checkpoint meta (injection_sites_config); a site absent there (warm
# start from a vanilla checkpoint) calibrates fresh below. Resumes use the
# PERSISTED ABSOLUTE vector and never re-resolve (loudness.json may have changed).
_auto_pending = dict(_auto_targets)
_dial_pending = dict(_dial_specs)
if resuming and (_auto_pending or _dial_pending):
    with open(os.path.join(checkpoint_dir, f"meta_{args.resume_from_step:06d}.json")) as f:
        _ckpt_sites = {c["name"]: c for c in json.load(f).get("injection_sites_config", [])}
    for _pending, _label in ((_auto_pending, "auto-gate"), (_dial_pending, "dial-gate")):
        for _name in list(_pending):
            _saved_gate = _ckpt_sites.get(_name, {}).get("gate")
            if isinstance(_saved_gate, (list, tuple)):
                _cfg = next(c for c in injection_cfgs if c.name == _name)
                _cfg.gate = [float(g) for g in _saved_gate]
                del _pending[_name]
                print0(f"[{_label}] {_name!r}: reusing the persisted ABSOLUTE gate from checkpoint meta (no re-resolve)")

# Compact-tokens mode (opt-in): a gemma-boundary-respecting tokenizer used for
# BOTH the training token stream (loader) and the runtime source alignment. OFF
# by default; when on it CHANGES the token stream (not baseline-comparable).
loader_tokenizer = tokenizer
_compact_nano_enc = _compact_gemma_encode = None
if args.compact_tokens:
    _runtime_specs = [s for s in injection_source_specs.values() if s.get("kind") == "probe-scores-runtime"]
    assert _runtime_specs, "--compact-tokens requires a 'probe-scores-runtime' source (gemma boundaries)"
    _gm = {s.get("gemma_model", "google/gemma-2-2b") for s in _runtime_specs}
    assert len(_gm) == 1, f"--compact-tokens needs one gemma model across runtime sources, got {_gm}"
    from nanochat.injection.sources import _default_gemma_encode
    from nanochat.injection.compact import CompactGemmaTokenizer
    _compact_gemma_encode = _default_gemma_encode(next(iter(_gm)))
    _compact_tok = CompactGemmaTokenizer(tokenizer, _compact_gemma_encode)
    _compact_nano_enc = _compact_tok.enc
    loader_tokenizer = _compact_tok
    print0("!" * 80)
    print0("--compact-tokens ON: nanochat tokens are cut at gemma boundaries. This CHANGES")
    print0("the training token stream (token-count inflation) — NOT comparable to a standard")
    print0("baseline. Alignment is 1:1 so mean/last policies coincide. (README: compact mode)")
    print0("!" * 80)

injection_sources = {}
for _name, _spec in injection_source_specs.items():
    _cfg = next(c for c in injection_cfgs if c.name == _name)
    _src = _open_injection_source(_name, _spec, _cfg, tokenizer, args.seed,
                                  nano_enc=_compact_nano_enc, gemma_encode=_compact_gemma_encode)
    assert _src.r == _cfg.r, f"site {_name!r}: source r={_src.r} != cfg r={_cfg.r}"
    if _spec.get("class"):
        # Hard guard: a configured custom class must be what actually got built
        # (the historical failure mode was 'class'/'kwargs' silently ignored).
        assert type(_src).__name__ == _spec["class"].rsplit(":", 1)[1], \
            f"site {_name!r}: configured class {_spec['class']!r} was not constructed (got {type(_src).__name__})"
    print0(f"[injection] site {_name!r}: source={type(_src).__name__} "
           f"(kind={_spec.get('kind')}, class={_spec.get('class') or '-'}) r={_src.r}")
    injection_sources[_name] = _src

# Gate calibration (auto + donor) — ENTIRELY at startup, BEFORE sites/prefill/
# step 1; sampled independently from the source (never the training buffer, so it
# cannot race prefill). Deterministic in (source, seed) => every DDP rank agrees;
# we assert bit-identical gate vectors across ranks. Provenance is persisted in
# checkpoint meta so resumes reuse the vector and never rescore.
_gate_provenance = {}   # name -> calibration meta (saved to checkpoint)


def _ddp_all_gather_hash(h):
    """Gather every rank's gate-hash string (None => single process)."""
    if not (ddp and ddp_world_size > 1 and is_ddp_initialized()):
        return None
    import torch.distributed as dist
    out = [None] * ddp_world_size
    dist.all_gather_object(out, h)
    return out


def _load_loudness(loc):
    """loudness.json from a local file, a local dir, or an HF dataset repo id."""
    if os.path.isfile(loc):
        with open(loc) as f:
            return json.load(f)
    if os.path.isdir(loc):
        with open(os.path.join(loc, "loudness.json")) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(loc, "loudness.json", repo_type="dataset")) as f:
        return json.load(f)


for _name, _target in _auto_pending.items():
    _cfg = next(c for c in injection_cfgs if c.name == _name)
    _gate_vec, _cal = calibrate_auto_gate(injection_sources[_name], target=_target,
                                          k=args.gate_k, seed=args.seed, min_docs=args.gate_min_docs)
    _cfg.gate = _gate_vec
    _cal["gate_hash"] = assert_gate_identical_across_ranks(_gate_vec, _name, print0, _ddp_all_gather_hash)
    _gate_provenance[_name] = {"mode": "auto", **_cal}
    print0(f"[auto-gate] {_name!r}: target={_target} active={_cal['n_active']}/{_cfg.r} "
           f"rms(gate)={_cal['rms_gate']:.4f} ({_cal['n_docs']} docs, {_cal['n_tokens']} tokens); "
           f"gate={[round(g, 4) for g in _gate_vec]}")

for _name, _spec in _dial_pending.items():
    _cfg = next(c for c in injection_cfgs if c.name == _name)
    _src = injection_sources[_name]
    _t_dial = time.time()
    # HARD requirement: a dial gate needs loudness.json + a source gemma layer;
    # discover/resolve raise actionable errors (naming abs:/--loudness-json).
    _loud, _loud_src = discover_loudness_json(_src, args.loudness_json or None, print0, _load_loudness)
    # For live/dynamic sources this actually runs the scorer over ~gate-k docs up
    # front (one-time startup cost, logged) — calibration NEVER runs lazily.
    _gate_vec, _cal = dial_gate_from_loudness(_src, _loud, dial=_spec.get("dial"),
                                              stat=_spec.get("stat"), k=args.gate_k,
                                              seed=args.seed, min_docs=args.gate_min_docs)
    _cfg.gate = _gate_vec
    _dur = time.time() - _t_dial
    _cal["gate_hash"] = assert_gate_identical_across_ranks(_gate_vec, _name, print0, _ddp_all_gather_hash)
    _cal.update({"loudness_source": _loud_src, "duration_s": round(_dur, 3)})
    _gate_provenance[_name] = _cal
    print0(f"[dial-gate] {_name!r}: dial={_cal['dial']:.4f}{' (donor:' + _spec['stat'] + ')' if _spec.get('stat') else ''} "
           f"L_ref={_cal['L_ref']:.4f} (layer {_cal['layer']}) -> abs rms(gate)={_cal['rms_gate']:.4f} "
           f"target={_cal['target_abs']:.4f} active={_cal['n_active']}/{_cfg.r} "
           f"({_cal['n_docs']} docs, {_cal['n_tokens']} tokens; {_dur:.1f}s startup scoring; {_loud_src})")

model.setup_injection_sites(injection_cfgs)
if args.activation_store:
    # Pin the frozen direction to the store's own P.npy when present (bit-exact
    # v1 forward even if the store's p_seed differs from the cfg default).
    _p_path = os.path.join(args.activation_store, "P.npy")
    if os.path.exists(_p_path):
        import numpy as np
        P = np.load(_p_path)  # (n_embd, r) float32 orthonormal
        assert P.shape[0] == model_config.n_embd, (P.shape, model_config.n_embd)
        assert P.shape[1] == injection_cfgs[0].r, (P.shape, injection_cfgs[0].r)
        with torch.no_grad():
            model.injection_sites["acts"].direction.copy_(
                torch.from_numpy(np.ascontiguousarray(P.T)).to(device))

# If we are resuming, overwrite the model parameters with those of the checkpoint
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    # Injected checkpoints carry injection_sites.* keys; a VANILLA checkpoint may
    # be warm-started (only site keys may be missing — they keep their fresh
    # init). assign=True replaces the Parameter objects, so re-stamp the
    # optimizability contract before setup_optimizer.
    missing, unexpected = model.load_state_dict(model_data, strict=False, assign=True)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    assert all(k.startswith("injection_sites.") for k in missing), f"missing non-injection keys: {missing}"
    if missing:
        print0(f"warm-start: checkpoint has no injection sites; {len(missing)} site tensors keep their fresh init")
    reassert_optimizability(model.injection_sites)
    for _name in list(_auto_targets) + list(_dial_specs):  # keep cfg (-> saved meta) in sync with the gate in the model
        _cfg = next(c for c in injection_cfgs if c.name == _name)
        _g = model.injection_sites[_name].gate.detach().float().cpu()
        _cfg.gate = _g.tolist() if _g.ndim else float(_g)
    del model_data # free up this memory after the copy

# Startup banner: source kind, shard coverage, resolved gates.
# NOTE: deliberately does NOT call len(_src) — for a runtime probe source without
# a hash index that walks EVERY shard's docs_*.jsonl, and it runs BEFORE
# _attach_prefetchers(), so multi-repo shard layouts (overflow repos via
# prefetch.repos/per_repo) 404 on the first shard outside the primary repo.
# The shard count is cheap and prefetch-independent.
for _name, _src in injection_sources.items():
    _cfg = next(c for c in injection_cfgs if c.name == _name)
    _nshards = len(getattr(_src, "shards", ()) or ())
    print0(f"injection site {_name!r}: source={getattr(_src, 'source_kind', type(_src).__name__)} "
           f"r={_src.r} after_block={_cfg.after_block} gate={_gate_str(_cfg.gate)} "
           f"trainable_direction={_cfg.trainable_direction} optim={_cfg.optim} "
           f"noise={float(injection_source_specs[_name].get('noise_sigma', 0.15))} shards={_nshards}")

# Optional ablation: disable the ResFormer value embeddings. We zero the value-embedding
# tables and freeze them, so they neither contribute to the forward (v = v + gate*0 == v)
# nor receive gradient updates. Done AFTER init_weights (and any resume load) so that,
# given the same --seed, every other parameter is bit-identical to the default run -- the
# only difference is whether value embeddings are learned. The ve_gate stays present but is
# a no-op (its output is multiplied by the zeroed value embedding).
if args.no_value_embeds:
    with torch.no_grad():
        for ve in model.value_embeds.values():
            ve.weight.zero_()
            ve.weight.requires_grad_(False)
    print0(f"--no-value-embeds: zeroed + froze {len(model.value_embeds)} value-embedding table(s)")

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Rolling shard prefetch (Amendment 3): stage score/parquet shards a small window
# ahead of consumption and delete consumed shards behind — minimal disk, no bulk
# pre-download. Config-driven per runtime source ("prefetch": {...}); a plain
# local score dir with everything present is a no-op passthrough (no prefetcher).
import threading as _threading
_prefetchers = []   # (name, ShardPrefetcher, auto_size) — re-sized after the prefill bandwidth measurement
def _attach_prefetchers():
    from nanochat.injection.prefetch import (ShardPrefetcher, make_hf_score_fetcher,
                                             make_local_deleter, repo_for_factory)
    for _name, _spec in injection_source_specs.items():
        pf = _spec.get("prefetch")
        if not pf:
            continue
        src = injection_sources[_name]
        assert hasattr(src, "prefetcher"), f"site {_name!r}: prefetch only supported for runtime probe sources"
        staging = pf["staging_dir"]
        climb = pf.get("climbmix_dir")
        repos, per_repo = pf.get("repos"), pf.get("per_repo")
        repo_for = (repo_for_factory(repos, per_repo) if repos
                    else (lambda sid, r=_spec["score_shards_dir_or_repo"]: r))
        fetch = make_hf_score_fetcher(staging, repo_for, climbmix_dir=climb)
        delete = make_local_deleter(staging, climbmix_dir=climb)
        order = sorted(src.shards)
        def _on_wait(sid, nm=_name):
            print0(f"[prefetch:{nm}] shard {sid} not staged yet — blocking (window behind; see starvation log)")
        # streams/ahead: explicit config wins; else the CLI default; else (both -1)
        # auto-size after prefill from the measured NIC bandwidth vs forecast consumption.
        cfg_streams = pf.get("streams", args.download_streams)
        auto_size = cfg_streams is None or int(cfg_streams) < 0
        init_streams = 2 if auto_size else int(cfg_streams)   # conservative start until sized
        # coord_dir + rank/world: every rank prefetches into the SAME staging dir;
        # deletion must key off the MIN frontier across ranks (a rank-local delete
        # frontier let fast ranks unlink shard files slow ranks were still reading
        # — FileNotFoundError/partial-JSON at prefill on 8xH100). The barrier
        # guarantees every rank has reset its .frontier_r{K} file before anyone
        # starts a worker (stale files from a crashed run would unblock deletion).
        p = ShardPrefetcher(order, fetch, delete_fn=delete, ahead=int(pf.get("ahead", 2)),
                            keep_behind=int(pf.get("keep_behind", 1)),
                            on_wait=_on_wait, on_delete=src.evict_shard, streams=init_streams,
                            coord_dir=staging, rank=ddp_rank, world_size=ddp_world_size)
        if ddp and ddp_world_size > 1 and is_ddp_initialized():
            import torch.distributed as dist
            dist.barrier()
        src._mm_lock = _threading.Lock()
        src.score_loc = staging          # per-shard reads now come from the local staging dir
        src.prefetcher = p.start()
        _prefetchers.append((_name, p, auto_size))
        print0(f"[prefetch:{_name}] rolling window ahead={p.ahead} keep_behind={p.keep_behind} "
               f"streams={p.streams}{' (auto)' if auto_size else ''} staging={staging} ({len(order)} shards)")
_attach_prefetchers()

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
# Ride-along loader: token path mirrors the stock best-fit loader 1:1, with
# per-site (B, T, r) activation tensors in lockstep. The val loader stays
# stock: eval is activations-off by design. loader_stats carries the Amendment-2
# staying-ahead counters (produce time/tokens, queue depth).
loader_stats = {}
# Video-player buffering: token-denominated targets, per rank. prefill warms the
# buffer before step 1; a dry buffer triggers a DDP-coordinated rebuffer.
_per_rank_step_tokens = max(1, total_batch_size // ddp_world_size)
prefill_tokens = args.prefill_tokens if args.prefill_tokens > 0 else 4 * _per_rank_step_tokens
rebuffer_tokens = args.rebuffer_tokens if args.rebuffer_tokens > 0 else max(1, prefill_tokens // 2)
dry_tokens = args.dry_tokens if args.dry_tokens > 0 else max(1, _per_rank_step_tokens)
buffer_max_tokens = args.buffer_max_tokens if args.buffer_max_tokens > 0 else 2 * prefill_tokens
buffer_max_tokens = max(buffer_max_tokens, prefill_tokens)
assert 0 <= dry_tokens < rebuffer_tokens <= prefill_tokens, (
    f"buffering needs dry < rebuffer <= prefill, got dry={dry_tokens} "
    f"rebuffer={rebuffer_tokens} prefill={prefill_tokens} (see --dry/--rebuffer/--prefill-tokens)")
buffer_ctrl, train_loader = acts_data_loader_buffered(
    loader_tokenizer, injection_sources, args.device_batch_size, args.max_seq_len, split="train",
    device=device, resume_state_dict=dataloader_resume_state_dict, lookup_workers=args.lookup_workers,
    stats=loader_stats, prefill_tokens=prefill_tokens, rebuffer_tokens=rebuffer_tokens,
    dry_tokens=dry_tokens, max_tokens=buffer_max_tokens)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)

# tty-aware buffering progress: carriage-return bar on a tty, plain periodic lines
# off it (rank 0 only). ``activity`` is the current shard file / align stage.
_buffer_tty = sys.stdout.isatty()
def _buffer_activity():
    for _nm, _p, _ in _prefetchers:
        s = _p.stats()
        return f"downloading shard {s['frontier'] + 1} / aligning"
    return "aligning"
def _buffer_progress(depth, target, rate):
    if not master_process:
        return
    _pct, _eta, msg = rebuffer_progress(depth, target, rate, activity=_buffer_activity())
    if _buffer_tty:
        sys.stdout.write("\r[buffering] " + msg + "   "); sys.stdout.flush()
    else:
        print(f"[buffering] {msg}", flush=True)

print0(f"[buffering] prefill={prefill_tokens:,} rebuffer={rebuffer_tokens:,} dry={dry_tokens:,} "
       f"max={buffer_max_tokens:,} tok (per rank); warming buffer before step 1...")
buffer_ctrl.wait_prefill(_buffer_progress)
if _buffer_tty and master_process:
    sys.stdout.write("\r" + " " * 100 + "\r"); sys.stdout.flush()
x, y, acts, dataloader_state_dict = next(train_loader) # first batch, drawn from the warmed buffer

# Free duty-cycle forecast (replaces the old sampling verdict): production rate was
# measured DURING prefill, so the forecast costs nothing extra. Consumption = this
# rank's total_batch_size/world tokens per --target-step-time seconds/step.
_prod_tok_s = buffer_ctrl.produce_rate()
_cons_tok_s = total_batch_size / ddp_world_size / max(args.target_step_time, 1e-9)
_ratio, _duty, _forecast = duty_cycle_forecast(_prod_tok_s, _cons_tok_s)
print0(f"[buffering] {_forecast} (source ~{_prod_tok_s:,.0f} tok/s vs this rank ~{_cons_tok_s:,.0f} tok/s "
       f"@ {args.target_step_time}s/step, workers={args.lookup_workers})")
if _ratio < 1.0 and args.starvation_abort:
    raise SystemExit(f"[buffering] --starvation-abort: forecast duty cycle ~{_duty*100:.0f}% (< 100%)")

# Prefetch auto-sizing: size download streams from the measured NIC bandwidth
# (mean fetch seconds over --shard-bytes) vs the seconds one shard lasts in
# training (--tokens-per-shard / consumption). Logs the arithmetic; raises streams.
def _size_prefetchers():
    if args.tokens_per_shard <= 0:
        return
    # global pace: ranks stride row groups WITHIN a shard, so every rank crosses
    # shard boundaries together (and each downloads every shard).
    cover_s = shard_cover_seconds(args.tokens_per_shard, _cons_tok_s, ddp_world_size)
    for _nm, _p, _auto in _prefetchers:
        if not _auto:
            continue
        _mfs = _p.stats().get("mean_fetch_seconds", 0.0)
        if _mfs <= 0:
            print0(f"[prefetch:{_nm}] no fetch timed during prefill; keeping streams={_p.streams}")
            continue
        _bw = args.shard_bytes / _mfs
        _streams, _ahead, _msg = size_prefetch_streams(args.shard_bytes, _bw, cover_s,
                                                       max_streams=8, min_ahead=_p.ahead)
        print0(f"[prefetch:{_nm}] {_msg}")
        _p.ahead = max(_p.ahead, _ahead)   # widen the window too, or the extra streams have nothing in-window to fetch
        _p.set_streams(_streams)
_size_prefetchers()

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Starvation monitor state (Amendment 2): rolling window of (blocked, step) times.
from collections import deque as _deque
_starv_wait = _deque(maxlen=args.starvation_window)
_starv_dt = _deque(maxlen=args.starvation_window)
_last_starv_warn = -10**9

# DDP-coordinated rebuffer: per-rank buffer-low flags all_reduce(MAX) at the step
# boundary, so ranks pause/refill/resume TOGETHER (independent per-rank stalls
# amplify at allreduce). A barrier after the refill keeps the resume in lockstep.
def _allreduce_max_int(v):
    t = torch.tensor([int(v)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())
def _rebuffer_if_dry(step):
    local_low = buffer_ctrl.buffer_low()
    if not coordinate_rebuffer(local_low, ddp_world_size, _allreduce_max_int):
        return
    if master_process:
        print0("!" * 80)
        print0(f"[buffering] activation buffer ran dry (step {step}) — pausing all ranks to "
               f"rebuffer to {rebuffer_tokens:,} tok")
        print0("!" * 80)
    buffer_ctrl.do_rebuffer(_buffer_progress)
    if is_ddp_initialized():
        dist.barrier()   # resume together
    if _buffer_tty and master_process:
        sys.stdout.write("\r" + " " * 100 + "\r"); sys.stdout.flush()
    print0(f"[buffering] resumed at buffer {buffer_ctrl.buffer_tokens():,} tok")

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: decide whether to write model weights and/or optimizer state this step.
    #   --save-every:     -1 = only final, 0 = never, N>0 = every N steps (+ final)
    #   --save-optimizer: 'every' (with each checkpoint) | 'final' (only last step) | 'never'
    if args.save_every == 0:
        save_model_now = False
        save_opt_now = False
    else:
        save_model_now = last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0)
        if args.save_optimizer == "never":
            save_opt_now = False
        elif args.save_optimizer == "final":
            save_opt_now = last_step
        else:  # "every"
            save_opt_now = save_model_now
        if save_opt_now and not save_model_now:
            save_model_now = True  # need a matching model checkpoint to use the optimizer state
    if save_model_now:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict() if save_opt_now else None, # optimizer state (optional)
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                # injection provenance: checkpoint_manager.build_model rebuilds
                # the sites from injection_sites_config so the keys load
                "injection_sites_config": [asdict(c) for c in injection_cfgs],
                # auto/donor gate provenance (target/stat/layer/rms/hash/loudness
                # source) — the calibrated vector rides in injection_sites_config
                # above; a resume reuses it and never rescores.
                "gate_calibration": _gate_provenance,
                "injection_source_specs": injection_source_specs,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
            compress=bool(args.compress_checkpoints),
            compress_level=args.checkpoint_compress_level,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # video-player rebuffer, DDP-coordinated (before consuming this step's batches)
    if step % max(1, args.buffer_check_every) == 0:
        _rebuffer_if_dry(step)

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    act_wait = 0.0  # time-blocked-waiting-on-activations this step (Amendment 2)
    for micro_step in range(grad_accum_steps):
        loss = model(x, y, acts=acts)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        _tw = time.time()
        x, y, acts, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        act_wait += time.time() - _tw
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            if not group.get('injection'):
                group["weight_decay"] = muon_weight_decay  # injection directions keep wd=0.0 (their scale is normalized away by the site)
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    # Staying-ahead (Amendment 2): blocked-time on activations + prefetch queue depth,
    # with a windowed starvation check (warn loudly, or hard-fail with --starvation-abort).
    _starv_wait.append(act_wait); _starv_dt.append(dt)
    q_depth = loader_stats.get("queue_depth", 0)
    buf_tok = loader_stats.get("buffer_tokens", 0)
    blocked_frac = act_wait / dt if dt > 0 else 0.0
    win_frac = (sum(_starv_dt) and sum(_starv_wait) / sum(_starv_dt)) or 0.0
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | act_wait: {act_wait*1000:.1f}ms ({blocked_frac*100:.0f}%) | qdepth: {q_depth} | buftok: {buf_tok:,} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if len(_starv_dt) >= min(args.starvation_window, 10) and win_frac > args.starvation_threshold:
        msg = (f"[staying-ahead] STARVATION: activation wait = {win_frac*100:.0f}% of step time over "
               f"last {len(_starv_dt)} steps (> {args.starvation_threshold*100:.0f}%); qdepth={q_depth}. "
               f"Source is not keeping up — raise --lookup-workers / widen the prefetch window.")
        if args.starvation_abort:
            raise SystemExit(msg)
        if step - _last_starv_warn >= args.starvation_window:
            print0("!" * 80); print0(msg); print0("!" * 80); _last_starv_warn = step
    if step % 100 == 0:
        log_data = {
            "step": step,
            "train/act_wait_ms": act_wait * 1000,
            "train/act_blocked_frac": win_frac,
            "train/act_queue_depth": q_depth,
            "train/buffer_tokens": buf_tok,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
