# Experiment 3 — weekday sphere (circle) manifold, gemma-L8-calibrated

Frozen injection direction whose 7 rows form a **1-sphere (circle)** in a 3-dim
subspace of nanochat's residual stream: a shared component `u0` plus a circular
component `(cos θ·u1 + sin θ·u2)`, giving all-positive inter-day cosines. The
shared/circular split (α, β) is least-squares-calibrated to gemma-2-2b's actual
measured weekday geometry at L8.

## 1. Measured gemma weekday geometry @ L8

Raw-space unit read directions `u_c = (W_c ⊘ nat_std) / ‖·‖` for the 7 weekday
concepts, replicating `attribution/verify_reconstruction.py`'s `Geometry` recipe
(L8 = axis-0 index **1** of the `[3, K, D]` arrays; store `layers == [6, 8, 14]`).
Concept store indices verified against `main_block_concepts`: friday=47, monday=48,
saturday=49, sunday=50, thursday=51, tuesday=52, wednesday=53.

Cosine-vs-**calendar**-circular-distance profile (means over 7 pairs each):

| circular distance | gemma mean cosine | within-distance std |
|---|---|---|
| 1 | **0.3381** | 0.0630 |
| 2 | **0.3049** | 0.0884 |
| 3 | **0.2522** | 0.0458 |

Full off-diagonal cosines span 0.202–0.487, **mean 0.298**. The profile is nearly
FLAT and all-positive: a large shared component with only a weak, noisy circular
ordering. (The single strongest pair is thursday–tuesday = 0.487, which is
calendar distance **2**, not 1 — the circular signal is genuinely weak.)

## 2. Fitted circle model

Model: `m_d = α·u0 + β·(cos θ_d·u1 + sin θ_d·u2)`, `θ_d = 2π·(calendar pos)/7`,
`u0,u1,u2` orthonormal (seeded random in R^768, seed **1337**, numpy fp64 QR).
Rows are unit-norm iff `α² + β² = 1`, giving the closed form
`cos(k) = ρ + (1-ρ)·cos(2πk/7)` with `ρ = α²`. LS fit of ρ to the 3-point profile:

- **ρ = 0.5199** (= α²), **α = 0.7210**, **β = 0.6929** (raw ρ = 0.5199, already in [0,1] — no clamp)
- fitted profile: **dist1 = 0.8192, dist2 = 0.4131, dist3 = 0.0874**

## 3. Does a shared-direction + circle model fit gemma? — NO (report honestly)

The residual is **LARGE**:

| distance | gemma | fitted circle | \|resid\| |
|---|---|---|---|
| 1 | 0.3381 | 0.8192 | **0.4811** |
| 2 | 0.3049 | 0.4131 | 0.1082 |
| 3 | 0.2522 | 0.0874 | 0.1649 |

- 3-point max \|resid\| = **0.4811**
- circle-model **R² = −15.12** against the 21 individual pairwise cosines (far
  worse than predicting the constant mean).

**Why:** the `cos(2πk/7)` regressors are widely spread (0.623, −0.222, −0.901),
so a rigid evenly-spaced circle *cannot* reproduce gemma's near-flat ~0.3 profile.
It preserves the qualitative story (all-positive, monotone-decreasing with
calendar distance) but massively exaggerates the distance dependence: it makes
adjacent days too similar (0.82 vs 0.34) and 3-apart days too dissimilar
(0.09 vs 0.25). Gemma's weekdays are closer to a **near-uniform positive clump /
simplex** than to a circle. The circle is imposed by design for exp3; the α/β
calibration is the closest a unit-norm circle gets to gemma's profile in LS.

This is the intended exp3 condition (a *realistic-looking* circle with positive
cosines), so it is not a blocker — but the geometry it injects is NOT a faithful
reproduction of gemma's actual weekday manifold. Flagging for interpretation.

## 4. Channel-order trap (tested)

Direction rows are in **STORE order** (friday first). Phases θ use **CALENDAR
position**. The two orders differ; the map is explicit and asserted in
`test_exp3.py`:

| store row | day | store idx | calendar pos | θ |
|---|---|---|---|---|
| 0 | friday | 47 | 4 | 2π·4/7 |
| 1 | monday | 48 | 0 | 0 |
| 2 | saturday | 49 | 5 | 2π·5/7 |
| 3 | sunday | 50 | 6 | 2π·6/7 |
| 4 | thursday | 51 | 3 | 2π·3/7 |
| 5 | tuesday | 52 | 1 | 2π·1/7 |
| 6 | wednesday | 53 | 2 | 2π·2/7 |

Constructed vs fitted-model profile agree to **1.7e-9**; rows unit-norm to
**2.8e-9**; every row's max off-diagonal cosine falls on its calendar
distance-1 neighbours (asserted).

## 5. Loading mechanism — MISSING, diff below (shared with exp4)

`InjectionCfg.direction_init` currently only supports `"orthonormal" | "zeros" |
"randn"` (`nanochat/injection/sites.py`), and `--activation-config` builds sites
purely from those dicts (no file path). An unknown value raises loudly (good — no
silent fallback). Add a generic `file:<path>` init. **~11 lines, frozen by default**
(`numpy` is already imported at the top of `sites.py`):

```diff
--- a/nanochat/injection/sites.py
+++ b/nanochat/injection/sites.py
@@ class InjectionCfg:
-    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn"
+    direction_init: str = "orthonormal"   # "orthonormal" | "zeros" | "randn" | "file:<path.npy|.npz>"
@@ class InjectionSite(nn.Module):
         if cfg.direction_init == "orthonormal":
             d0 = orthonormal_direction(cfg.r, n_embd, cfg.direction_seed)
         elif cfg.direction_init == "zeros":
             d0 = torch.zeros(cfg.r, n_embd)  # only sensible trainable (frozen zeros = dead site)
         elif cfg.direction_init == "randn":
             g = torch.Generator().manual_seed(cfg.direction_seed)
             d0 = torch.randn(cfg.r, n_embd, generator=g) / n_embd ** 0.5
+        elif cfg.direction_init.startswith("file:"):
+            # Frozen (r, n_embd) direction from a .npy/.npz (npz key "D", else its
+            # sole array). Rows verbatim; the site's z/rms(z) renorm supplies the
+            # per-token scale. Frozen unless trainable_direction=True. Path is
+            # resolved relative to CWD (launch from the nanochat repo root).
+            _p = cfg.direction_init[len("file:"):]
+            _l = np.load(_p)
+            _a = (_l["D"] if "D" in _l.files else _l[_l.files[0]]) if hasattr(_l, "files") else _l
+            d0 = torch.from_numpy(np.ascontiguousarray(_a)).float()
+            assert tuple(d0.shape) == (cfg.r, n_embd), \
+                f"file direction {_p!r}: shape {tuple(d0.shape)} != (r={cfg.r}, n_embd={n_embd})"
         else:
             raise ValueError(f"unknown direction_init {cfg.direction_init!r}")
         self.direction = nn.Parameter(d0, requires_grad=bool(cfg.trainable_direction))
```

`direction_sphere.npz` stores the direction under key **`D`** ([7, 768] float32,
store-channel order), so the loader picks it up directly.

## 6. wandb project — MISSING override (shared with all runs)

`scripts/injection_train.py` hardcodes `wandb.init(project="nanochat", ...)`
(line ~161). To log exp3 into the **"weekday-geometry"** project, parametrize it
(env `WANDB_PROJECT` won't win against an explicit `project=` arg):

```diff
--- a/scripts/injection_train.py
+++ b/scripts/injection_train.py
@@ parser.add_argument("--run", ...)
 parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
+parser.add_argument("--wandb-project", type=str, default="nanochat", help="wandb project name")
@@ wandb logging init
-wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
+wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=args.wandb_project, name=args.run, config=user_config)
```

Then add `--wandb-project weekday-geometry` to the launch command. Without the
diff the run still logs (to project "nanochat", run name "exp3-sphere").

## 7. Exact launch command

Three framework hooks are prerequisites (all shared / applied once by the
consolidator):
- **DIFF 1** — `_open_injection_source` custom-`class`/`kwargs` hook
  (`NOTES_2_trainable.md`, exp2). MANDATORY: wires `WeekdayProbeScoreSource` +
  the `present_z=2.0` realism threshold. Without it the threshold is silently
  dropped and weekday z injects on every covered token.
- **DIFF 3** — the §5 `sites.py` `file:` `direction_init` loader. MANDATORY for
  exp3/exp4 (exp3 fails loud at startup without it).
- **DIFF 2** — the §6 `--wandb-project` flag. Only needed to log into the
  `weekday-geometry` project (drop the flag to fall back to `nanochat`).

`exp3_config.json`'s **source block is byte-identical to `exp2_config.json`**
(`WeekdayProbeScoreSource`, `present_z=2.0`, `kaushikreddyxyz/climbmix-scored`
`shards 0-184` + `-overflow..-overflow-7` prefetch, layer 8, weekday concepts in
store order, `align_policy=mean`, `noise_sigma=0`). The **site block differs from
exp2 only in the direction**: `trainable_direction=false`,
`direction_init="file:runs/weekdays/direction_sphere.npz"`; `after_block=3`,
`gate="abs:0.0273"`, `r=7`, site name `weekdays` are shared.

From the **nanochat repo root** (`.../oracle-encodings/nanochat`):

```bash
bash runs/weekdays/exp3_sphere.sh                 # NPROC / DEPTH / … env-overridable
SMOKE=1 bash runs/weekdays/exp3_sphere.sh         # single-process, a few steps, nothing saved
```

which runs (mirrors `exp2_trainable.sh`; training budget uses injection_train's
compute-optimal defaults):

```bash
torchrun --standalone --nproc_per_node=$NPROC -m scripts.injection_train -- \
  --activation-config runs/weekdays/exp3_config.json \
  --depth 12 --no-value-embeds --seed 1337 \
  --device-batch-size 16 --model-tag weekday_exp3_sphere \
  --save-every 2000 --save-optimizer final --compress-checkpoints 1 \
  --eval-every 250 --core-metric-every 2000 --sample-every 2000 \
  --lookup-workers 8 --wandb-project weekday-geometry --run exp3-sphere
```

Post-run, push checkpoints to `kaushikreddyxyz/weekday-geometry-d12` folder
`sphere/` (see `scripts/push_to_hf.py`; the launcher's tail comment has the exact
command).

## Files (all under runs/weekdays/)
- `build_manifold.py` — measures gemma L8 geometry, fits α/β, builds D, emits artifacts (deterministic, CPU, $0).
- `direction_sphere.npz` — `D` [7,768] f32 (store order) + metadata (α, β, ρ, profiles, seed, calendar map).
- `manifold_validation.json` — gemma & constructed 7×7 cosine matrices, profile comparison, max errors.
- `test_exp3.py` — plain-assert structural tests (all pass).
- `exp3_config.json`, `exp3_sphere.sh` — launch wiring.
