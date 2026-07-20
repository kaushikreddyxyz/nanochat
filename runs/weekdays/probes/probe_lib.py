"""Weekday view of the concept-general probing library (runs/lib/probe_lib.py):
re-exports its generic machinery pinned to the 7-weekday site (name 'weekdays',
after_block 3, cols 47..53). Behavior-identical to the pre-extraction module.
Used by train_probes.py / test_probes.py (import probe_lib), both unchanged.
"""
import importlib.util
import os
import sys

_LIB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "lib"))


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_LIB, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Loaded by absolute path (an installed PyPI 'runs' shadows the local runs/ dir);
# unique module name so it never collides with this file's own 'probe_lib' name.
_lib = _load("_weekday_probe_core", "probe_lib.py")

# Generic machinery, re-exported byte-for-byte from the package.
Moments = _lib.Moments
RowBuffer = _lib.RowBuffer
ProbeSpec = _lib.ProbeSpec
pack_batches = _lib.pack_batches
predict = _lib.predict
r2_from_moments = _lib.r2_from_moments
solve_ridge = _lib.solve_ridge
save_condition = _lib.save_condition
LAMBDA_GRID = _lib.LAMBDA_GRID

_FAMILY = "weekdays"
SITE_NAME = "weekdays"
AFTER_BLOCK = 3
D_MODEL = 768
STORE = list(_lib.concept_registry.get_family(_FAMILY).store_order)  # cols 47..53
R = len(STORE)


def _n_embd(model):
    return int(getattr(getattr(model, "config", None), "n_embd", D_MODEL))


def ResidCapture(model, after_block=AFTER_BLOCK, site_name=SITE_NAME):
    """Weekday-pinned residual capture (delegates to runs/lib ResidCapture). d_model
    is read from the model so a tiny test GPT and the d12 model both work."""
    spec = ProbeSpec(site_name=site_name, after_block=after_block,
                     d_model=_n_embd(model), concepts=tuple(STORE))
    return _lib.ResidCapture(model, spec)


def run_condition(model, docs, bos_id, *, loudness_scale, use_acts, device, forward_metrics,
                  heldout_every=10, max_rows=32, max_tokens=65536, lambda_grid=LAMBDA_GRID,
                  log=print):
    """Weekday-pinned condition runner: builds a weekday ProbeSpec, then delegates to
    runs/lib run_condition (identical ridge/R^2/readout math)."""
    spec = ProbeSpec.from_family(_FAMILY, AFTER_BLOCK, _n_embd(model),
                                 site_name=SITE_NAME, lambda_grid=tuple(lambda_grid))
    return _lib.run_condition(model, docs, bos_id, spec, loudness_scale=loudness_scale,
                              use_acts=use_acts, device=device, forward_metrics=forward_metrics,
                              heldout_every=heldout_every, max_rows=max_rows,
                              max_tokens=max_tokens, log=log)
