"""Feature injection for nanochat. Two families (see README.md here):

1. Geometric manifolds (``inject.py``): a frozen additive feature that is a
   pure function of the token id (ring/line/sphere/helix in reserved residual
   dims), added before the trunk. Smoke test: ``python -m nanochat.injection.smoke``.
2. Contextual activation injection (``sites.py`` + ``sources.py`` +
   ``activation_dataloader.py``): per-token-occurrence activations from a
   pluggable ActivationSource, added after a chosen block through
   gate/activation/direction injection sites. Trained by
   ``scripts/injection_train.py``.

``activation_dataloader`` is deliberately NOT imported here (it pulls in
``nanochat.dataloader`` -> pyarrow); import it explicitly. Same for ``align``
(precompute-side tokenizer bridge).
"""
from nanochat.injection.inject import (
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
from nanochat.injection.sites import (
    InjectionCfg,
    InjectionSite,
    build_sites,
    optimizer_param_split,
    orthonormal_direction,
    reassert_optimizability,
    sites_by_block,
)
from nanochat.injection.sources import (
    ActivationSource,
    FnSource,
    ProbeScoreSource,
    QwenEncoderSource,
    build_structured_activations,
    doc_hash,
    make_orthonormal_P,
    open_store,
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
    "ActivationSource",
    "FnSource",
    "ProbeScoreSource",
    "QwenEncoderSource",
    "open_store",
    "build_structured_activations",
    "doc_hash",
    "make_orthonormal_P",
]
