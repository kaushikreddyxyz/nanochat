"""Thin binding from the eval suite to the shared registry's seasons family: wires
runs/lib onto sys.path once and re-exports the family's two orderings plus cycle
arithmetic. Ordering data lives ONLY in runs/lib/concepts.py.
Import: from seasons_concepts import STORE_ORDER, CYCLE_ORDER, R, season_plus
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
_LIB_DIR = os.path.join(_REPO_ROOT, "runs", "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

import concepts  # noqa: E402

FAMILY = concepts.get_family("seasons")

# Activation-channel order (name-sorted store columns) — a channel index means THIS.
STORE_ORDER = list(FAMILY.store_order)
STORE_COLUMNS = list(FAMILY.store_index)
# Canonical seasonal cycle — every next/previous/k-step answer is read off THIS list.
CYCLE_ORDER = list(FAMILY.cycle_order)
R = FAMILY.r


def season_plus(season, k):
    """The season k steps after ``season`` around the cycle (k may be negative)."""
    return CYCLE_ORDER[(FAMILY.cycle_position(season) + k) % R]
