#!/usr/bin/env python3
"""Emit the weekdays 7-point circle direction (direction_sphere.npz) + the orthogonal
control + validation json, via runs/lib/manifold.py. CPU-only, deterministic.
Run: python runs/weekdays/build_directions.py

The committed npz/json here predate runs/lib and carry the OLD key spelling
(gemma_*/constructed_*/calendar_map, weekday_store_order). Rerunning this rewrites them
in the library schema (donor_*/built_*/cycle_map). D is bit-identical either way, and D
is the only key the site loader and the checkpoints ever read.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))   # nanochat repo root
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

import manifold  # noqa: E402


def main():
    manifold.build_family("weekdays", HERE, layer=8, seed=manifold.DEFAULT_SEED,
                          n_embd=manifold.DEFAULT_N_EMBD, orthogonal=True)


if __name__ == "__main__":
    main()
