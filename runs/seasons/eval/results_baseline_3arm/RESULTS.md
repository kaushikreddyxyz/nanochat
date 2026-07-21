# Baseline d12 — 3-arm seasonal-ablation reference eval

Reference capability numbers for the **frozen d12 baseline** (`nanochat-d12-injections/baseline/`,
step 2520, CORE ≈0.1449), to be compared against the injection-trained arms.

## Arms
| arm (file key) | what |
|---|---|
| `off` | baseline, plain (no intervention) |
| `ablate_L0` | project out the 4 seasonal DoM directions **at the most-salient layer L6** (that layer's dirs) |
| `ablate_all` | project them out **at every block, each block using its own layer's dirs** |

Ablation = `x ← x − (x·Q̂)Q̂`, Q̂ = orthonormal basis (rank 4/block) of the seasonal DoM **write**
directions `W_dom ⊙ nat_std` for {autumn, spring, summer, winter}, from `baseline/probes/`.
MOST_SALIENT_LAYER = **L6**, chosen as the layer maximizing both the mean (0.907) and min (0.901)
DoM AUROC over the 4 seasons. Only seasonal directions are ablated. No injection, no gemma.

## Results  (completion = 4-option MC accuracy, chance = 0.25)
| arm | seasons_v2 (n=632) | colors_v2 control (n=632) | CORE (t=22) | seasons_v1 (n=354) |
|---|---|---|---|---|
| baseline (off) | 0.4130 | 0.5744 | 0.1457 | 0.3362 |
| ablate @ L6    | 0.3402 | 0.4937 | 0.1457 | 0.3277 |
| ablate @ all   | 0.3038 | 0.4320 | 0.1457 | 0.2910 |

## Findings
1. **Seasons are causally load-bearing on the DoM directions** — accuracy falls toward chance
   (0.413 → 0.304), monotonically worse from L6-only to all-layers. Per-tier, T0_copy/T1_assoc/
   T2_member/T3_hop collapse to exactly 0.250 (chance) once ablated (see `seasons/baseline_*.json`
   `completion.by_category`).
2. **CORE is untouched** (0.1457 across all arms; matches the baseline's known CORE) — the
   intervention does not damage broad language capability.
3. **The colors control is NOT clean** — ablating *seasonal* directions also lowers *colors* MC
   (0.574 → 0.432). Since CORE is flat, this is specific to concept-MC, not generic capacity loss:
   the seasonal DoM directions are **entangled** with color/concept representations in this small
   d12 (a shared broad-concept component). Relative to above-chance signal, seasons lose ~67%,
   colors ~44% — more seasonal than not, but far from specific.

## Caveat / next
A **random-direction control** (ablate 4 random dirs/layer) would separate "seasonal↔color
entanglement" from "removing any 4 dims/layer breaks concept-MC". Not yet run.

## Provenance
Produced by `runs/seasons/eval/run_baseline_3arm.sh` → `runs/lib/eval/run_evals.py`
(`--baseline-ablate-dir baseline/probes --baseline-ablate-layer 6
--baseline-ablate-concepts autumn spring summer winter`, arms `off ablate_L0 ablate_all`).
Per-arm JSONs hold per-tier/per-category breakdowns; `summary.json` is the flat metric table.
