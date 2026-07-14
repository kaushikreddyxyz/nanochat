# Experiment 4 — orthogonal (null) weekday geometry

## Scientific rationale (one paragraph)

Experiment 4 is the null-geometry control of the weekday study. Its frozen
injection direction is 7 **mutually orthogonal** unit rows in R^768 (one per
weekday channel), so every pair of days has **zero cosine similarity**: there is
no shared "weekday-ness" axis the days project onto, and no cyclic adjacency
structure (Monday is no closer to Tuesday than to Saturday). This is a geometry a
transformer would *not* naturally instill for a cyclic concept family — the
neural-geometry literature (e.g. arXiv 2602.15029) finds real learned
representations of cyclic families share a common subspace and lie on a
ring/circle. Experiment 3 injects exactly that realistic shared-direction + circle
manifold; Experiment 4 injects its opposite. Holding the source activations, gate,
site, and every other knob identical across the two runs, the *only* difference is
the direction's geometry, so any downstream divergence isolates the causal effect
of manifold shape (realistic-cyclic vs orthogonal-null).

## Direction mechanism — two interchangeable paths

The 7 orthogonal rows come from `nanochat.injection.sites.orthonormal_direction(r=7,
n_embd=768, seed=1337)`. Verified numerically: `D @ D.T` is identity to fp32
precision — **max |off-diagonal cosine| = 2.4e-7**, **max |row-norm − 1| =
6.0e-8** (see `orthogonal_validation.json`). A QR of seeded fp64 randn (what that
function does internally) gives exact orthonormality; that is precisely the null
manifold we want, so the framework default *is* the artifact.

- **Path A — seed route (RECOMMENDED, ZERO source changes).** `exp4_config.json`
  sets `direction_init: "orthonormal"`, `direction_seed: 1337`,
  `trainable_direction: false`. `InjectionSite.__init__` builds the rows from the
  seed at startup; the direction is frozen (`requires_grad=False`). Nothing new
  is needed in the framework — this is the stock path and what
  `exp4_orthogonal.sh` launches.
- **Path B — file route (for symmetry with exp3).** `direction_orthogonal.npz`
  (key `D`, float32 `[7, 768]`) is **bit-identical** to the seed route's matrix
  (`test_exp4.py::test_npz_bit_exact_roundtrip_and_matches_seed_route` asserts
  `np.array_equal`). Exp3 needs a `direction_init: "file:<path>"` mechanism for
  its custom circle manifold (that geometry has no seed that produces it). **If
  the consolidator lands that mechanism, exp4's npz slots straight in** — swap the
  site's `direction_init` to `"file:runs/weekdays/direction_orthogonal.npz"` with
  no behavioral change. Exp4 does **not** require it; Path A stands alone.

## Files (all under `runs/weekdays/`)

| file | purpose |
|---|---|
| `make_direction_orthogonal.py` | regenerates the npz + validation json from the framework generator |
| `direction_orthogonal.npz` | frozen direction, key `D`, float32 `[7, 768]` (bit-identical to seed 1337) |
| `orthogonal_validation.json` | 7×7 cosine matrix, row norms, max|off-diag|=2.4e-7 |
| `exp4_config.json` | `--activation-config` JSON: site (direction, gate, geometry) + shared source stub |
| `exp4_orthogonal.sh` | launch command mirroring the pinned shared config |
| `test_exp4.py` | plain-assert checks (all 5 pass on CPU) |

`exp4_config.json` is **IDENTICAL to `exp2_config.json` except
`trainable_direction: false`** (exp2 learns from the same orthonormal init;
exp4 stays frozen). Its **`sources` block is the canonical shared block** (owned
by exp2): `class: runs.weekdays.weekday_source:WeekdayProbeScoreSource` +
`kwargs: {present_z: 2.0}`, which applies the realism threshold (per token: keep
the 7 weekday z-scores only if `max_c z_c >= 2.0`, else an exact-zero row) on top
of the stock runtime probe path (layer 8, climbmix-scored + overflows prefetch,
the 7 weekday concepts in store order 47..53, noise 0). The `class`/`kwargs`
fields **require exp2's `_open_injection_source` hook (their NOTES DIFF 1)**;
without it the subclass — and thus the realism threshold — is silently NOT used.
Exp4 owns only `sites[0]` (the frozen orthogonal direction); the source block is
shared verbatim across exp2/3/4.

## Pinned shared config as realized here

depth 12 (n_embd 768) · `--no-value-embeds` · default tokenization · gemma layer 8
scores from `kaushikreddyxyz/climbmix-scored` (+overflows, ops-owned prefetch) ·
weekday channels in store order (friday 47, monday 48, saturday 49, sunday 50,
thursday 51, tuesday 52, wednesday 53) · realism threshold `max_c z_c >= 2.0` else
zero row (shared wiring) · gate `abs:0.0273` · noise σ 0 · after_block 3 · injection
from step 0 · wandb project `weekday-geometry`, run `exp4-orthogonal` · HF
checkpoints `kaushikreddyxyz/weekday-geometry-d12` under `orthogonal/`.

## Exact launch command

```bash
# from the repo root
export WANDB_PROJECT="weekday-geometry"     # needs Diff 1 below (upstream hardcodes "nanochat")
export HF_HUB_DISABLE_XET=1
python -m scripts.injection_train -- \
  --run exp4-orthogonal \
  --model-tag weekday-geometry-d12-orthogonal \
  --depth 12 \
  --no-value-embeds \
  --activation-config runs/weekdays/exp4_config.json \
  --noise-sigma 0
```

(Or just `bash runs/weekdays/exp4_orthogonal.sh`. Prepend
`torchrun --nproc_per_node=8` for multi-GPU.) The gate `abs:0.0273`, after_block 3,
r=7, and frozen orthonormal direction all live in `exp4_config.json`: on the
`--activation-config` path the gate is read from the site dict and the `--gate` CLI
flag is ignored. Checkpoints land in
`$NANOCHAT_BASE_DIR/base_checkpoints/weekday-geometry-d12-orthogonal`; push to
`kaushikreddyxyz/weekday-geometry-d12/orthogonal/` as a separate step.

## Proposed source diffs (NOT applied — per the hard rule)

**Shared source-wiring hook (exp2's DIFF 1, not re-specified here):** the config's
`sources.weekdays.class`/`kwargs` fields need a small generic branch in
`scripts/injection_train.py::_open_injection_source` that imports the named class
and passes `kwargs` — see exp2's `NOTES_2_trainable.md`. Exp4 depends on the same
hook (all injected runs share it); without it the realism threshold no-ops
silently.

### Diff 1 (SHARED across all 4 runs, REQUIRED for the wandb project)

`scripts/injection_train.py` hardcodes `project="nanochat"`, so `WANDB_PROJECT`
is ignored. Make it env-overridable (minimal, `os` is already imported at line 21):

```diff
--- a/scripts/injection_train.py
+++ b/scripts/injection_train.py
@@ -161,1 +161,1 @@
-wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
+wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=os.environ.get("WANDB_PROJECT", "nanochat"), name=args.run, config=user_config)
```

Without it the run still trains; it just logs to the `nanochat` project. This diff
is shared — exp2/exp3 need it too.

### Diff 2 (OPTIONAL — exp3 owns it; exp4 does NOT need it)

A `direction_init: "file:<path>"` branch lets a site load a frozen direction from
an .npy/.npz (exp3's circle manifold requires this; exp4's npz would then slot in
via Path B). Provided for completeness; exp4 runs without it.

```diff
--- a/nanochat/injection/sites.py
+++ b/nanochat/injection/sites.py
@@ -67,6 +67,13 @@ class InjectionSite(nn.Module):
         if cfg.direction_init == "orthonormal":
             d0 = orthonormal_direction(cfg.r, n_embd, cfg.direction_seed)
+        elif isinstance(cfg.direction_init, str) and cfg.direction_init.startswith("file:"):
+            _arr = np.load(cfg.direction_init[len("file:"):])
+            if hasattr(_arr, "files"):          # .npz -> take the first array
+                _arr = _arr[_arr.files[0]]
+            d0 = torch.from_numpy(np.ascontiguousarray(_arr, dtype=np.float32))
+            assert d0.shape == (cfg.r, n_embd), \
+                f"direction file shape {tuple(d0.shape)} != (r={cfg.r}, n_embd={n_embd})"
         elif cfg.direction_init == "zeros":
             d0 = torch.zeros(cfg.r, n_embd)  # only sensible trainable (frozen zeros = dead site)
```

(`numpy as np` is already imported at the top of `sites.py`.) To use it, set the
site's `direction_init` to `"file:runs/weekdays/direction_orthogonal.npz"`.
