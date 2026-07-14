# NOTES — injection-on-vs-off eval harness (foundation)

`harness.py` is the foundation the two sibling suites (evalset/CORE wiring; causal
protocol) import. New files only under `runs/weekdays/eval/`; no existing source
edited; no commit.

## Interface (PINNED — delivered exactly)

| symbol | signature | notes |
|---|---|---|
| `load_model` | `(arm, device, hf_repo="kaushikreddyxyz/weekday-geometry-d12", step=2520) -> (model, meta)` | downloads + builds; sites rebuilt from meta; baseline has NO site |
| `attach_site` | `(model, direction, gate=0.0273, after_block=3, name="weekdays") -> InjectionSite` | bolt an inference-time site onto the BASELINE (causal control); `direction` = `[7,768]` array or `"sphere"`/`"orthogonal"` (pulls the arm npz) |
| `GemmaScorer` | `.score(texts) -> list[np.ndarray [n_gemma,7] fp32 z]` | gemma-2-2b L8 residual → 7 frozen weekday ridge probes → step-2 z. Offsets accessors: `.gemma_encode(text) -> (ids, offsets)` (single text; sibling's first probe), `.gemma_offsets(texts)`, `.score_with_offsets(texts)`, `.tokenizer` (gemma fast tok) |
| `build_acts` | `(text, nano_ids, gemma_z, offsets, threshold=2.0, policy="mean", nano_enc=None) -> np.ndarray [T,7]` | overlap-mean align + realism threshold, EXACTLY like training |
| `forward_metrics` | `(model, ids, acts=None, gate_scale=1.0, transform=None) -> dict` | per-token CE + logits; on/off via `gate_scale` |
| `ce_report` | `(per_token_ce, acts) -> dict` | dilution-aware CE decomposition |

Extra (auxiliary, non-breaking): `set_nano_tokenizer(enc)`, `WEEKDAY_CONCEPTS`,
constants (`SITE_NAME`, `R`, `AFTER_BLOCK`, `GATE`, `N_EMBD`, `GEMMA_LAYER`).
`build_acts` gained an optional trailing `nano_enc=` kwarg (defaults to the
module tokenizer) — the pinned positional args are unchanged.

### The on/off invariant (do not break)
On-vs-off is **always** `gate_scale=1.0` vs `gate_scale=0.0` through the SAME acts
code path — never `acts=None` vs `acts=…`. `gate_scale` temporarily multiplies
every site's gate in place (`_scaled_gates` context manager) and restores it on
exit. `gate=0` is an exact framework no-op (verified: `test_harness` asserts
`gate_scale=0` logits are **bit-identical** to the `acts=None` forward on a d=64
model that HAS a site). Sites are never optimized during eval — the gate carries
`_never_optimize` and sits in no optimizer group — so in-place scaling under
`no_grad` (then forward under `inference_mode`) is safe.

## Checkpoint-loading findings (verified offline against the real metas)

- **`checkpoint_manager.build_model` rebuilds sites cleanly** from
  `meta["injection_sites_config"]` (its L133–142) — `setup_injection_sites` +
  strict `assign` load + `reassert_optimizability`. **But we do NOT call it**, for
  two reasons, and replicate its model-building steps instead:
  1. **rustbpe coupling.** `checkpoint_manager` does `from nanochat.tokenizer
     import get_tokenizer` at module load, and `nanochat/tokenizer.py` does
     `import rustbpe` at module load. rustbpe is unavailable off-pod, so
     **`checkpoint_manager` cannot even be imported** locally. The tokenizer is
     an injected dependency here (we never need `get_tokenizer`), so `load_model`
     inlines the three pure helpers (`_load_tensorfile`,
     `_patch_missing_config_keys`, `_patch_missing_keys`) byte-for-byte and
     imports only `nanochat.gpt` (rustbpe-free). Net: **importing `harness` pulls
     in no rustbpe** — sibling agents can import it anywhere.
  2. **The `file:` direction (sphere arm).** `sphere/meta_002520.json` persists
     `direction_init: "file:runs/weekdays/direction_sphere.npz"`. In
     `InjectionSite.__init__` the `file:` branch does `np.load(<relative path>)`
     resolved from the **launch CWD** during `setup_injection_sites` — it needs
     CWD == repo root **and** the npz present, else `setup` raises
     `FileNotFoundError` (verified: (a) a `file:` init fails without the npz).
     Since the site's direction is **immediately overwritten** by the strict
     `assign` load of the checkpoint's saved `injection_sites.weekdays.direction`
     (strict=True guarantees the key is present), the file contents are
     irrelevant — only the shape matters. `load_model` therefore **rewrites any
     `direction_init: "file:*"` → `"zeros"`** before `setup_injection_sites`,
     making loading self-contained (no CWD / npz dependency). Verified: (b) the
     rewrite yields the correct site (r=7, after_block=3, gate=0.0273, direction
     `(7,768)`, `gate._never_optimize=True`). The `trainable`/`orthogonal` arms
     use `orthonormal` init (unchanged).
- **Gate persists as a SCALAR `0.0273`** (not a vector) for all three injected
  arms. `gate_scale` multiplies it; `gate_scale=0` → exact `0.0`.
- **Baseline does NOT confuse `attach_site`.** `baseline/meta_002520.json` has
  `injection_sites_config: null`, so `load_model("baseline", …)` returns a model
  with **no** `injection_sites` attribute. `attach_site` asserts no existing
  `"weekdays"` site and attaches a fresh one — no confusion. (Confirmed the three
  injected metas carry the config and baseline does not.)
- HF layout: `…/weekday-geometry-d12/<arm>/model_002520.pt.gz` (+ `meta_…json`,
  `optim_…rank{0..7}.pt.gz`). `_download` tries `.pt.gz` then `.pt`.
  Model config: `n_layer=12, n_embd=768, vocab_size=32768, sequence_len=2048`.

## No source diffs required
The harness reuses everything and needs **no edit to existing source**.

- **`build_acts` reuses the vendored alignment + threshold verbatim.** It builds a
  store-free `WeekdayProbeScoreSource` via `object.__new__` and sets only the
  attributes `_align_and_gather` touches (`nano_enc`, `gemma_encode`,
  `align_policy`, `r`, `present_z`), feeding the precomputed gemma tokenization
  through `gemma_encode` (so no gemma model/tokenizer re-runs). This calls the
  SAME `_RuntimeProbeBase._align_and_gather` overlap-mean/last code and the SAME
  `WeekdayProbeScoreSource` threshold that training ran — no reimplementation.

- **Optional ergonomic diff (not needed, offered):** a public
  `WeekdayProbeScoreSource.aligner_only(nano_enc, gemma_encode, *, present_z,
  align_policy, r)` classmethod would replace the `object.__new__` shim with a
  named constructor. Purely cosmetic; skip unless the team wants it.

## Device / dtype policy
- **Model:** matches `build_model` — kept as saved (**bf16**) on CUDA; bf16→float
  on cpu/mps. Forward casts activations to `COMPUTE_DTYPE`; `logits` are returned
  fp32 by `GPT.forward`; **per-token CE is accumulated in fp32**
  (`F.cross_entropy` on fp32 logits, `reduction="none"`).
- **Gemma (GemmaScorer):** eager attention **MANDATORY** (sdpa drops gemma-2
  softcapping); **bf16** on CUDA / fp16 on mps / fp32 on cpu; BOS prepended then
  its row dropped; residual read at `hidden_states[L+1]` (L=8 → index 9); norms +
  probe math in fp64, output cast fp32. Windows tiled at 2048 like the scorer.

## Acts packing / BOS convention (for the eval driver)
`build_acts` returns the **body** activations `[T,7]`, `T = len(nano_ids)`
(exact-zero rows where no weekday ≥ threshold). Training prepends a **BOS zero
row** per doc and the loader aligns acts to INPUTS (`acts = row_act[:, :-1]`). So
when feeding a single doc `ids = [bos] + nano_ids` to `forward_metrics`, the
driver builds `acts[B,T,7]` with **row t = the activation for input token t** and
a **zero BOS row** at position 0 (`np.concatenate([zeros(1,7), body])`). Confirmed
end-to-end on the tiny model: off logits are bit-identical to vanilla, on logits
differ, `ce_report` buckets partition the valid positions.

`forward_metrics` indexes `per_token_ce[b,t]` to the **predicted** token t
(`-log p(ids[t] | ids[:t])`, position 0 = NaN). `ce_report` marks token t
injected iff `acts[b,t]` is nonzero, so `ce_injected` = CE of predicting injected
tokens, `ce_after_injected` = CE of the next token, `ce_other` = the rest
(injected tokens are ~2–3% of the corpus, so the conditional means are the
signal).

## Sibling-interface reconciliations (so `run_evals.py` / `causal.py` need no edit)
- **`build_acts` without `nano_enc`.** The siblings call `build_acts(text,
  nano_ids, gemma_z, offsets, threshold=…)` and neither pass `nano_enc` nor call
  `set_nano_tokenizer`. `build_acts` resolves the nano tokenizer as: explicit
  `nano_enc=` → module default (`set_nano_tokenizer`) → **pod fallback**
  (`get_tokenizer().enc`, imported lazily inside the call). Tests always pass an
  explicit fake, so the rustbpe fallback never fires off-pod (verified). A
  `RustBPETokenizer` wrapper passed as `nano_enc` is auto-unwrapped to its
  tiktoken `.enc`.
- **Gemma offsets accessor.** `run_evals._gemma_z_and_offsets` probes for
  `gemma_encode` first; `GemmaScorer.gemma_encode(text) -> (ids, offsets)` is
  provided (same convention as `RuntimeProbeScoreSource.gemma_encode`), plus
  `.tokenizer` as a further fallback.
- **CE key.** `forward_metrics` returns the key `"per_token_ce"`, which
  `run_evals._per_token_ce` accepts first.

## GPU-side TODOs (structured so the pod run just works)
1. **Nanochat tokenizer (injected).** Either rely on the `build_acts` pod
   fallback above, or (to avoid re-constructing) inject once:
   `from nanochat.tokenizer import get_tokenizer; harness.set_nano_tokenizer(get_tokenizer().enc)`
   (`.enc` is the tiktoken Encoding exposing `encode_ordinary` +
   `decode_single_token_bytes`). Or pass `nano_enc=` to `build_acts` per call.
2. **GemmaScorer prerequisites:** gemma-2-2b is gated (HF login / `HF_TOKEN`);
   `probe_set_arrays.npz` + `probe_set.json` must be present — auto-discovered via
   `$ORACLE_ATTR_OUT` → `<repo>/../attribution/out` → `./attribution/out`
   (superproject `attribution/out` verified locally). The store's
   `columns.json` + `corpus_stats.json` are fetched from
   `kaushikreddyxyz/climbmix-scored` at init (needs network). Constant-loading was
   validated offline (mocked store): `W (7,2304)`, `b/mu2/std2 (7,)`,
   `nat_mean/nat_std (2304,)`, weekday cols `[47..53]`, layer 8, no gemma built.
3. **Live-vs-store gap:** GemmaScorer computes z **live** (pre-quantization);
   training injected the store's **int8** z (~0.03σ resolution). Equal up to
   quantization, not bit-identical. For an on/off pass on the TRAINING corpus,
   reading the store directly (`RuntimeProbeScoreSource`) reproduces the injected
   z exactly; GemmaScorer is for **fresh eval texts** not in the store.
4. **Checkpoints:** ~448 MB `.pt.gz` per arm from the HF model repo (needs access).
5. **CE at scale:** `forward_metrics` returns full `logits[B,T,V]` (V=32768) on
   CPU fp32 — fine for eval batch sizes; slice before large B·T.

## Test table (`test_harness.py`, CPU, no network / rustbpe / gemma)
| test | asserts |
|---|---|
| `test_gate_scale_zero_is_exact_noop_and_restores` | `gate_scale=0` logits **bit-identical** to `acts=None`; gate restored exactly after `gate_scale∈{0,0.5}`; `gate_scale=1` w/ nonzero acts changes logits |
| `test_build_acts_alignment_broadcast_and_mean` | one gemma token broadcasts to nested nano tokens; a multi-gemma nano span pools by MEAN; `policy="last"` keeps the rightmost — reuses the vendored aligner |
| `test_build_acts_threshold_exact_zero` | sub-threshold nano rows are **exact** zero; supra-threshold rows survive |
| `test_ce_report_buckets` | injected / after / other counts + means correct; buckets partition the valid positions |

Run: `python runs/weekdays/eval/test_harness.py` (4/4 pass).
