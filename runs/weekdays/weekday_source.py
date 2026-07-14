"""Experiment-side activation source for the weekday-geometry runs.

A THIN subclass of ``nanochat.injection.sources.RuntimeProbeScoreSource`` that
adds the two experiment-specific behaviours the shared weekday config needs, on
top of the stock runtime probe-score path (dequant + standardize + overlap
alignment, all inherited unchanged):

  1. **7-channel weekday subset.** The parent already subsets columns for free
     via its ``concepts=`` constructor arg — ``_init_layout`` looks each name up
     in ``columns.json`` order to build ``col_idx``. We do NOT re-implement the
     slicing; the config passes the 7 weekday concept names (name-sorted store
     order: friday, monday, saturday, sunday, thursday, tuesday, wednesday ==
     store column indices 47..53) and the parent produces r=7, col_idx =
     [47,48,49,50,51,52,53], reading gemma layer 8 (store axis-1 index 1).

  2. **Realism threshold (present_z, default 2.0).** After the parent aligns the
     gemma probe scores onto the nanochat token grid (overlap MEAN by default),
     each per-nanochat-token row is set to EXACT ZERO unless the max over the 7
     weekday channels of the (standardized) z-score is >= present_z — i.e. unless
     at least one weekday concept is present in its positive tail. A framework
     zero-row is an exact injection no-op (the site renormalizes only nonzero
     rows), so this makes the injection fire only on genuinely weekday-ish tokens.
     Rows the parent already returned as exact zero (unmapped / no covering gemma
     token) stay zero (max 0 < 2.0). The threshold is applied POST-pool, matching
     "after alignment pooling".

Everything else (positional (shard,row) join, noise, stats, prefetch hook,
None/drift -> exact-zero contract) is inherited verbatim. This file is NEW and
edits no framework source; wiring it into ``scripts/injection_train.py`` needs a
tiny generic hook in ``_open_injection_source`` (see NOTES_2_trainable.md).
"""
from nanochat.injection.sources import RuntimeProbeScoreSource

# Canonical weekday channel order == kaushikreddyxyz/climbmix-scored name-sorted
# store column order (indices 47..53). ALL weekday-geometry experiments must use
# this exact order so the r=7 activation vectors are comparable across runs.
WEEKDAY_CONCEPTS = [
    "friday",    # store col 47
    "monday",    # store col 48
    "saturday",  # store col 49
    "sunday",    # store col 50
    "thursday",  # store col 51
    "tuesday",   # store col 52
    "wednesday", # store col 53
]


class WeekdayProbeScoreSource(RuntimeProbeScoreSource):
    """RuntimeProbeScoreSource + post-alignment realism threshold.

    Extra kwarg (passed via the source spec's ``kwargs`` block):
      present_z : float (default 2.0) — a nanochat-token row survives only if
                  max over its 7 weekday channels >= present_z, else it is zeroed.
    Column subsetting is handled entirely by the parent (pass ``concepts=
    WEEKDAY_CONCEPTS``); this class does NOT touch col_idx.
    """

    def __init__(self, *args, present_z=2.0, **kwargs):
        self.present_z = float(present_z)
        super().__init__(*args, **kwargs)

    def _align_and_gather(self, text, n_tokens, z_gemma):
        # Parent does the full gemma->nanochat overlap alignment (mean/last) and
        # returns (n_tokens, r) standardized z, or None (unknown/drift).
        out = super()._align_and_gather(text, n_tokens, z_gemma)
        if out is None:
            return None
        # Realism threshold: keep a row only if some weekday channel is in its
        # positive tail (>= present_z sigma); zero every other row EXACTLY so the
        # site injects a strict no-op there. Rows already exact-zero (unmapped)
        # have max 0 < present_z and stay zero. ">=" so the boundary is inclusive.
        keep = out.max(axis=1) >= self.present_z
        out[~keep] = 0.0
        return out
