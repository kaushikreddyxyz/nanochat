# Trainable weekday-direction: learned geometry

Structural analysis of the **trainable** arm's 7x768 injection direction `D`
after 2,520 AdamW steps (orthonormal init, seed 1337, gate 0.0273 scalar).
Checkpoint `kaushikreddyxyz/weekday-geometry-d12` `trainable/model_002520.pt.gz`.
Rows are in **store order** (friday, monday, saturday, sunday, thursday,
tuesday, wednesday = climbmix cols 47-53); every calendar-ordered / circular
quantity remaps store-row -> calendar position explicitly (`CAL_ROWS =
[1,5,6,4,0,2,3]`, asserted). References: init (== `direction_orthogonal.npz`,
reproduced to 0.0), sphere arm (`direction_sphere.npz`), gemma-2b L8 measured
(`manifold_validation.json`).

Run: `python3 runs/weekdays/direction_analysis/analyze_direction.py` (CPU, $0,
deterministic). Every number is in `direction_report.json`.

## Headline
Off-diag cosine **mean +0.022** (std 0.238, range [-0.465, +0.486]) — the 7
learned rows are, on average, **mutually orthogonal**, unlike gemma (mean +0.30,
all-positive) or the sphere arm (circular). Each row **rotated ~90 deg from its
init** (per-row |cos| mean 0.029) and the whole 7-dim row-subspace rotated to
**near-orthogonal** to init (principal angles 80-89 deg, overlap 0.012). Norms
grew from unit to ~85-111. Variance is **spread** (centered PC1 36%, k2 55%, k3
72%; effective rank 5.1). The calendar-ring signal is **faint**: cosine falls
0.065 -> 0.027 -> -0.025 across calendar distance 1/2/3 (gemma 0.34/0.30/0.25),
and free-phase order does **not** recover any clean calendar ring.

## Figures
- **fig1_cosine_matrices.png** — 7x7 inter-day cosine in calendar order (Mon..Sun),
  three panels: learned / gemma-L8-measured / sphere-arm. Learned is a mixed
  near-zero field (blues and light reds, mean +0.02); gemma is uniformly positive
  (~0.2-0.5); sphere is the calibrated circular checkerboard. *The optimizer did
  not reproduce gemma's all-positive weekday-cluster geometry.*
- **fig2_norms_drift.png** — (a) learned row norms 85-111 (init was unit, dashed),
  Tue smallest / Sat largest; (b) per-row cos(learned, init) ~0 for every day;
  (c) 7 principal angles between the learned and init row-subspaces, all 80-89 deg.
  *Both the rows and the subspace they span rotated almost fully away from init.*
- **fig3_pca_scree.png** — per-PC and cumulative explained variance, raw (left) and
  mean-centered (right). Centered k1 36%, k2 55%, k3 72%; no dominant component.
  *No low-rank (2-3D) structure — variance is distributed across ~5 effective dims.*
- **fig4_circle_projections.png** — mean-centered rows on PC1-PC2 and PC2-PC3,
  days connected in calendar order (Mon->Sun). The calendar path is tangled, not a
  convex ring. *No calendar-ordered ring in the top principal planes.*
- **fig5_circular_profile.png** — mean inter-day cosine vs calendar circular
  distance (1/2/3) for learned / gemma / sphere, +/-1 std. Learned is a faint
  downward slope hugging zero (error bars cross 0); gemma a gentle positive slope;
  sphere a steep 0.82->0.09. *A weak monotone calendar gradient exists but is ~5x
  smaller than gemma's and not significant against pair spread.*
- **fig6_free_phases.png** — free per-day phases (best shared-removed 2D plane,
  left) vs the ideal uniform calendar ring (right). Calendar colors are scrambled
  around the learned circle; recovered order by calendar pos is [5,3,1,0,6,4,2].
  *Free-phase order matches no clean ring (forward, reverse, or star winding).*

## One-line interpretation
Training did **not** sculpt gemma's positive-cosine cluster or a calendar ring;
it drove the 7 directions to a fresh, near-orthogonal, higher-rank subspace with
only a faint residual calendar gradient — the loudness-renorm site absorbs
absolute scale, so the optimizer had no pressure to keep the rows aligned or
low-rank.
