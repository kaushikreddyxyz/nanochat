#!/usr/bin/env python3
"""Experiment 4 (orthogonal / null geometry) — emit the frozen injection direction.

The 7 direction rows (one per weekday channel) are 7 MUTUALLY ORTHOGONAL unit
vectors in R^768: zero cosine similarity between every pair of days. This is the
deliberately UNREALISTIC manifold — no shared "weekday-ness" axis, no cyclic
adjacency structure — the null geometry against experiment 3's realistic
shared-direction + circle manifold.

We reuse the FRAMEWORK's own generator (nanochat.injection.sites.orthonormal_direction)
so the emitted .npz is BIT-IDENTICAL to what direction_init="orthonormal" with the
same direction_seed builds in-model. That means the two mechanisms (seed route vs
file route) are interchangeable — see NOTES_4_orthogonal.md.

Outputs (under runs/weekdays/):
  direction_orthogonal.npz    -> array 'D', float32 [7, 768]
  orthogonal_validation.json  -> 7x7 cosine matrix, row norms, max |off-diag|
"""
import json
import os

import numpy as np
import torch

from nanochat.injection.sites import orthonormal_direction

R = 7            # weekday channels (friday..wednesday, store order)
N_EMBD = 768     # depth=12 -> n_embd = 12 * 64 = 768
SEED = 1337      # pinned; == InjectionCfg.direction_seed default and store p_seed default

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    D = orthonormal_direction(R, N_EMBD, SEED)          # torch.float32 [7, 768]
    D_np = np.ascontiguousarray(D.numpy(), dtype=np.float32)

    npz_path = os.path.join(HERE, "direction_orthogonal.npz")
    np.savez(npz_path, D=D_np)

    # Validation: 7x7 cosine matrix (rows are ~unit norm, so cosine ~= Gram matrix).
    norms = np.linalg.norm(D_np, axis=1)
    unit = D_np / norms[:, None]
    cos = unit @ unit.T
    off = cos - np.eye(R, dtype=cos.dtype)
    max_off_diag = float(np.abs(off).max())

    validation = {
        "experiment": "exp4-orthogonal",
        "description": "7 mutually orthogonal unit rows in R^768 (null / unrealistic weekday geometry)",
        "shape": [R, N_EMBD],
        "dtype": "float32",
        "direction_seed": SEED,
        "generator": "nanochat.injection.sites.orthonormal_direction",
        "weekday_channel_order": [
            "friday", "monday", "saturday", "sunday",
            "thursday", "tuesday", "wednesday",
        ],
        "row_norms": [float(x) for x in norms],
        "cosine_matrix": [[float(v) for v in row] for row in cos],
        "max_abs_off_diagonal_cosine": max_off_diag,
        "max_abs_row_norm_error": float(np.abs(norms - 1.0).max()),
    }
    json_path = os.path.join(HERE, "orthogonal_validation.json")
    with open(json_path, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"wrote {npz_path}  (D {D_np.shape} {D_np.dtype})")
    print(f"wrote {json_path}")
    print(f"max |off-diag cosine| = {max_off_diag:.3e}")
    print(f"max |row-norm - 1|     = {validation['max_abs_row_norm_error']:.3e}")


if __name__ == "__main__":
    main()
