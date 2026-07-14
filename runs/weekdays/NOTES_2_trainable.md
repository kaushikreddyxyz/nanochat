# Weekday-geometry Experiment 2 — trainable direction (implementation notes)

Files in `runs/weekdays/` (all NEW; no existing source edited):

- `weekday_source.py` — `WeekdayProbeScoreSource`, a thin subclass of
  `RuntimeProbeScoreSource` (7-weekday subset via the parent's `concepts=`, plus
  the `present_z=2.0` realism threshold applied post-alignment).
- `exp2_config.json` — the `--activation-config` payload (one trainable site
  `weekdays`, r=7, after_block=3, gate `abs:0.0273`; runtime probe source, gemma
  L8, shards 0-184, prefetch block).
- `exp2_trainable.sh` — full `scripts.injection_train` launch command.
- `test_exp2.py` — plain-assert CPU tests (subset/layer-axis, threshold, config
  round-trip). **All pass** (`python3 runs/weekdays/test_exp2.py`).

---

## The wiring answer: can injection_train use the custom source WITHOUT a source edit?

**No — one tiny generic edit is required (DIFF 1).** `injection_train._open_injection_source`
(`scripts/injection_train.py` ~lines 218-238) dispatches source construction on a
hardcoded set of `kind`s: `"probe-scores-runtime"` → `RuntimeProbeScoreSource`,
everything else → `open_store(...)` (store kinds). There is **no import-path /
custom-class hook**, so a JSON config cannot name an arbitrary `ActivationSource`
subclass. The `probe-scores-live`/`FnSource` paths are programmatic (not
JSON-wireable) and don't help.

The **column subset alone** needs no edit — `RuntimeProbeScoreSource(concepts=[...])`
already slices columns for free (`_init_layout` builds `col_idx` from
`columns.json` order). It is **only the realism threshold** that requires a custom
class, hence the edit.

Two options for the consolidator:

- **Preferred: apply DIFF 1** (below) — a 6-line, fully backward-compatible hook
  that lets any `RuntimeProbeScoreSource` subclass be named from the config via
  `"class": "module:Class"` + `"kwargs": {...}`. Then `exp2_config.json` works
  as-is and `WeekdayProbeScoreSource` (threshold included) is used.
- **Alternative (no source edit at all):** drop `"class"`/`"kwargs"` from the
  source spec and accept a plain `RuntimeProbeScoreSource` with the 7-column
  subset but **NO threshold** — i.e. weekday z injected on *every* covered token.
  This is a *different experiment* (no realism gating) and is **not** what Exp 2
  specifies, so DIFF 1 is the intended path.

> **Silent-degradation warning:** without DIFF 1, `spec.get("class")` /
> `spec.get("kwargs")` are simply ignored (unknown JSON keys), so the run would
> quietly train with the un-thresholded plain source. The launch script and this
> note flag DIFF 1 as **mandatory** before launch.

---

## DIFF 1 (MANDATORY) — generic custom-source hook in `_open_injection_source`

`scripts/injection_train.py`, inside the `if kind == "probe-scores-runtime":`
branch. Adds (a) an optional `"class"` override of the source class — accepting
BOTH a **file path** (`path.py:Class`, collision-proof) and a **dotted module**
(`pkg.mod:Class`) — and (b) pass-through of extra `"kwargs"`. No behaviour change
when neither key is present (existing configs unaffected).

```diff
     kind = spec.get("kind")
     if kind == "probe-scores-runtime":
         from scripts.precompute_activations import parse_shard_range
         sh = spec["shards"]
         shards = parse_shard_range(sh) if isinstance(sh, str) else [int(s) for s in sh]
-        return RuntimeProbeScoreSource(
+        src_cls = RuntimeProbeScoreSource
+        if spec.get("class"):                      # experiment-side ActivationSource subclass
+            import importlib
+            _target, _cls = spec["class"].rsplit(":", 1)
+            if _target.endswith(".py") or "/" in _target:    # file path (resolved from launch CWD = repo root)
+                import importlib.util, os
+                _p = _target if os.path.isabs(_target) else os.path.join(os.getcwd(), _target)
+                _s = importlib.util.spec_from_file_location(f"_inj_src_{_cls}", _p)
+                _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
+            else:                                            # dotted module path (needs the module importable)
+                _m = importlib.import_module(_target)
+            src_cls = getattr(_m, _cls)
+            assert issubclass(src_cls, RuntimeProbeScoreSource), \
+                f"source 'class' {spec['class']!r} must subclass RuntimeProbeScoreSource"
+        return src_cls(
             spec["score_shards_dir_or_repo"], shards, layer=int(spec.get("layer", 8)),
             nano_enc=nano_enc or tok.enc, gemma_encode=gemma_encode, concepts=spec.get("concepts"),
             gemma_model=spec.get("gemma_model", "google/gemma-2-2b"),
             align_policy=spec.get("align_policy", "mean"),
             climbmix_dir=spec.get("climbmix_dir"), index_path=spec.get("index_path"),
             build_hash_index=bool(spec.get("build_hash_index", False)),
-            noise_sigma=float(spec.get("noise_sigma", 0.15)), seed=seed, name=name)
+            noise_sigma=float(spec.get("noise_sigma", 0.15)), seed=seed, name=name,
+            **spec.get("kwargs", {}))
```

`weekday_source.py` imports only `nanochat.injection.sources`, so there is no
import cycle either way. `exp2_config.json` uses the **file-path** form
(`runs/weekdays/weekday_source.py:WeekdayProbeScoreSource`), resolved from the
launch CWD which every `runs/weekdays/*.sh` sets to the repo root.

> **Import-form caveat (verified on this machine):** the dotted form
> `runs.weekdays.weekday_source:...` — used by `exp4_config.json` — relies on
> `runs/` being importable as a PEP-420 namespace package. That FAILS if a regular
> package named `runs` is installed (there is a PyPI `runs` in this dev env's
> site-packages, and a regular package shadows a namespace package regardless of
> `sys.path` order). The **file-path form has no such failure mode**. Recommend the
> consolidator standardize all weekday configs on the file-path form (or add
> `runs/__init__.py` + `runs/weekdays/__init__.py` — new files I could not create
> here, being outside `runs/weekdays/`). The DIFF above supports both so no config
> is broken; only the dotted form is env-fragile.

## DIFF 2 (needed only for the pinned wandb project) — `--wandb-project` flag

`wandb.init` hardcodes `project="nanochat"` (`scripts/injection_train.py:161`), and
an explicit `project=` **overrides** the `WANDB_PROJECT` env var, so the env alone
cannot redirect the run. To land Exp 2 in project `weekday-geometry`:

```diff
@@ CLI args (near --run) @@
 parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
+parser.add_argument("--wandb-project", type=str, default="nanochat", help="wandb project name")
@@ wandb init (line ~161) @@
-wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
+wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=args.wandb_project, name=args.run, config=user_config)
```

`exp2_trainable.sh` passes `--wandb-project weekday-geometry`. Without DIFF 2 the
flag errors on argparse (loud, not silent) — remove it to fall back to project
`nanochat` if the consolidator prefers to defer this diff. This diff benefits all
4 weekday-geometry runs.

---

## How the source wiring works (once DIFF 1 is in)

1. `injection_train` reads `--activation-config`, builds `InjectionCfg(**site)` for
   the one site and keeps `sources["weekdays"]` as the source spec.
2. Gate: `classify_gate_spec("abs:0.0273")` → `("abs", 0.0273)` → the site gate is
   the **absolute scalar 0.0273** (raw residual-RMS fraction). Because it is
   `abs:`, **no** loudness.json fetch, **no** `sample_activation_stats` calibration,
   **no** startup scoring — resolution is instant and needs no gemma weights.
3. `_open_injection_source` sees `kind == "probe-scores-runtime"` and (DIFF 1)
   `class == "runs.weekdays.weekday_source:WeekdayProbeScoreSource"`, imports it,
   and constructs it with `concepts=[7 weekdays]`, `layer=8`, `align_policy="mean"`,
   `noise_sigma=0.0`, and `**{"present_z": 2.0}`.
4. The parent produces `r=7`, `col_idx=[47,48,49,50,51,52,53]`, `li=1` (gemma L8 ==
   store axis-1 index 1). Per doc it dequantizes+standardizes those 7 columns and
   overlap-mean-aligns them onto the nanochat grid; the subclass then zeroes every
   row whose max weekday z < 2.0.
5. `model.setup_injection_sites([cfg])` builds one `InjectionSite` after block 3
   with a **trainable** orthonormal `(7, 768)` direction (`direction_seed=1337`).

## Trainable-direction optimizer evidence (cited, NOT modified)

The trainable direction IS picked up by the optimizer — verified in-repo, no edit:

- `nanochat/injection/sites.py::optimizer_param_split` returns each site's
  `direction` **iff `direction.requires_grad`**, splitting by `cfg.optim`
  (`adamw`/`muon`). Gates (`_never_optimize`) and frozen directions are excluded.
- `nanochat/gpt.py::setup_optimizer` (line ~417) calls
  `optimizer_param_split(self.injection_sites)` → `inj_adamw, inj_muon`, and adds
  them as dedicated AdamW / Muon groups tagged `injection=True` with
  `weight_decay=0.0` (lines ~448 / ~454). The param-count assertion at line ~418
  guarantees every parameter lands in exactly one group.
- Tests: `tests/test_injection_sites.py:138-152` (`optimizer_param_split` returns
  the two trainable directions, not the frozen one, and gates carry
  `_never_optimize`), and `:359-362` (`m.setup_optimizer()` with a trainable site
  builds a working optimizer). `injection_train` runs `optimizer.step()` each step
  (line ~1025), and the wd scheduler skips `injection`-tagged groups (line ~1013).

Exp 2 uses `"optim": "adamw"` (framework default) → the direction joins the AdamW
injection group (wd=0.0, betas (0.8, 0.995)). Change to `"muon"` in the site cfg to
route it through Muon instead; the task did not specify, so the default is used.

## Exact launch command

```bash
# from repo root, after DIFF 1 (+ DIFF 2 for the wandb project) are applied:
bash runs/weekdays/exp2_trainable.sh
# == torchrun --standalone --nproc_per_node=8 -m scripts.injection_train -- \
#      --activation-config runs/weekdays/exp2_config.json \
#      --depth 12 --no-value-embeds --seed 1337 --device-batch-size 16 \
#      --model-tag weekday_exp2_trainable \
#      --save-every 2000 --save-optimizer final --compress-checkpoints 1 \
#      --eval-every 250 --core-metric-every 2000 --sample-every 2000 \
#      --lookup-workers 8 --wandb-project weekday-geometry --run exp2-trainable
# SMOKE first:  SMOKE=1 bash runs/weekdays/exp2_trainable.sh
```

Prereqs on the pod: `$NANOCHAT_BASE_DIR/tokenizer` (baseline tokenizer) and
`$NANOCHAT_BASE_DIR/base_data_climbmix` (`python -m nanochat.dataset`); HF auth for
the gated gemma-2 tokenizer and the `climbmix-scored*` dataset repos;
`HF_HUB_DISABLE_XET=1` (set by the script). Checkpoints land locally under
`base_checkpoints/weekday_exp2_trainable`; mirror the final one to
`kaushikreddyxyz/weekday-geometry-d12` folder `trainable/` as a separate push step
(injection_train does not push to HF).

## Open questions / things to verify before/at launch

- **gemma_model = tokenizer only.** The source uses `gemma_model` solely for the
  fast tokenizer that produces char offsets for alignment. The gemma-2 family
  (2b/9b) shares one tokenizer, so the default `google/gemma-2-2b` is correct even
  though the probe scores are gemma-2-9b **L8** — no 9b weights are downloaded.
  (If a preflight ever shows n_gemma drift, that would indicate a tokenizer
  mismatch, not this.)
- **Prefetch repo layout.** `prefetch.repos` (base + `-overflow..-overflow-7`) and
  `per_repo: 25` follow the README example + the scorer's known 25-shards/repo
  layout, covering shards 0-199 ⊇ 0-184. Confirm the exact overflow repo names and
  per-repo counts against the actual scorer output before launch; adjust `repos`/
  `per_repo` (or drop them to pull all shards from one repo) if they differ.
  `streams: 4` is set explicitly (no `--tokens-per-shard` auto-sizing); raise if
  the starvation monitor warns.
- **Realism-threshold sanity.** With `present_z=2.0` on standardized z that
  overlap-MEAN pooling shrinks, the fraction of surviving (nonzero) tokens may be
  small; watch the source `.stats()` / the smoke banner and the `act_wait` /
  buffer logs. If too sparse, `present_z` is the single knob (in the config
  `kwargs`) — but keep it 2.0 across all 4 runs for comparability unless the whole
  series changes it.
- **DIFF 1 is a hard pre-launch gate** (else the threshold silently no-ops). A
  quick guard: after wiring, the startup banner line for site `weekdays` should
  show `source=WeekdayProbeScoreSource r=7 after_block=3`.
- **Cross-run shard-range discrepancy (consolidator decision).** This task pins
  `shards 0-184` for exp2 (and exp4 matches). `exp3_config.json` instead uses
  `shards 300-399` (its `_comment` explicitly defers the shard list + score repos
  to "the exp2 agent / consolidator"). Pick ONE training-doc shard set across all
  4 runs so they train on the same corpus slice and stay comparable — I followed
  the exp2 spec (0-184); reconcile exp3 before launch.
- **Import-form consistency (consolidator decision).** exp2 uses the file-path
  `"class"` form; exp4 uses the dotted-module form. Both work under DIFF 1, but the
  dotted form is env-fragile (see the caveat under DIFF 1). Recommend standardizing
  on the file-path form.
