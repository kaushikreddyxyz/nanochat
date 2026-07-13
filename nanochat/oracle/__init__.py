"""Oracle-feature injection for nanochat.

Sibling of ``modular_addition/oracle``. Two feature families:

1. *Geometric manifolds* (``inject.py``): a frozen additive feature that is a
   pure function of the token id — token ids placed on a ring, line, sphere, or
   helix in a few reserved residual dimensions, injected right before the
   transformer trunk (see the hook in ``nanochat.gpt.GPT.forward``).

2. *Contextual coord injection* (``injections.py`` + ``coords_store.py`` +
   ``coord_dataloader.py``): per-token-occurrence activations (e.g. frozen-Qwen
   probe-score coords precomputed by ``scripts/precompute_coords.py``) added
   after a chosen block through gate/activation/direction injection sites.
   See README.md in this directory.

``coord_dataloader`` is deliberately NOT imported here: it pulls in
``nanochat.dataloader`` (pyarrow). Import it explicitly as
``nanochat.oracle.coord_dataloader`` where needed. Same for ``align``
(precompute-side tokenizer bridge).

Public API:
    attach_oracle, detach_oracle, freeze_params
    make_table_oracle, make_manifold_oracle, make_ring_oracle
    ring_coords, line_coords, sphere_coords, helix_coords
    InjectionCfg, InjectionSite, build_sites, sites_by_block,
    optimizer_param_split, reassert_optimizability, orthonormal_direction,
    FnActivation
    CoordSource, build_coords, make_orthonormal_P, doc_hash
"""
from nanochat.oracle.inject import (
    attach_oracle,
    detach_oracle,
    freeze_params,
    helix_coords,
    line_coords,
    make_manifold_oracle,
    make_ring_oracle,
    make_table_oracle,
    ring_coords,
    sphere_coords,
)
from nanochat.oracle.injections import (
    FnActivation,
    InjectionCfg,
    InjectionSite,
    build_sites,
    optimizer_param_split,
    orthonormal_direction,
    reassert_optimizability,
    sites_by_block,
)
from nanochat.oracle.coords_store import (
    CoordSource,
    build_coords,
    doc_hash,
    make_orthonormal_P,
)

__all__ = [
    "attach_oracle",
    "detach_oracle",
    "freeze_params",
    "make_table_oracle",
    "make_manifold_oracle",
    "make_ring_oracle",
    "ring_coords",
    "line_coords",
    "sphere_coords",
    "helix_coords",
    "InjectionCfg",
    "InjectionSite",
    "build_sites",
    "sites_by_block",
    "optimizer_param_split",
    "reassert_optimizability",
    "orthonormal_direction",
    "FnActivation",
    "CoordSource",
    "build_coords",
    "doc_hash",
    "make_orthonormal_P",
]
