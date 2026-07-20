"""Every runs/ test lives here; this puts the modules under test on sys.path so they
import by bare name (the collision-proof file-path convention the injection configs use
— an installed PyPI ``runs`` package shadows the local runs/ dir). Family item banks and
eval sets carry family-prefixed basenames, so one flat sys.path never mixes two families.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUNS = os.path.join(REPO, "runs")
LIB = os.path.join(RUNS, "lib")
LIB_EVAL = os.path.join(LIB, "eval")
WEEKDAYS = os.path.join(RUNS, "weekdays")
SEASONS = os.path.join(RUNS, "seasons")

for _p in (REPO, LIB, LIB_EVAL, os.path.join(WEEKDAYS, "eval"),
           os.path.join(WEEKDAYS, "probes"), os.path.join(SEASONS, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
