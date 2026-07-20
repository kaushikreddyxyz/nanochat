#!/usr/bin/env python3
"""Emit the seasons 4-point circle direction (direction_sphere.npz) + the orthogonal
control + validation json, via runs/lib/manifold.py. CPU-only, deterministic.
Run: python runs/seasons/build_directions.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))   # nanochat repo root
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

import manifold  # noqa: E402


def main():
    manifold.build_family("seasons", HERE, layer=8, seed=manifold.DEFAULT_SEED,
                          n_embd=manifold.DEFAULT_N_EMBD, orthogonal=True)


if __name__ == "__main__":
    main()
