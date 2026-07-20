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
    assert_loudness_identical_across_ranks,
    build_sites,
    calibrate_dose_gate,
    classify_loudness_spec,
    discover_loudness_json,
    donor_source_layer_concepts,
    initial_direction,
    log_realized_loudness,
    loudness_vector_hash,
    optimizer_param_split,
    orthonormal_direction,
    realized_loudness_report,
    reassert_optimizability,
    sites_by_block,
    validate_donor_concepts,
)
from nanochat.injection.sources import (
    ActivationSource,
    FnSource,
    LiveProbeScoreSource,
    ProbeScoreSource,
    QwenEncoderSource,
    RuntimeProbeScoreSource,
    build_structured_activations,
    doc_hash,
    hash_seeded_noise,
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
    "initial_direction",
    "calibrate_dose_gate",
    "classify_loudness_spec",
    "donor_source_layer_concepts",
    "validate_donor_concepts",
    "discover_loudness_json",
    "realized_loudness_report",
    "log_realized_loudness",
    "loudness_vector_hash",
    "assert_loudness_identical_across_ranks",
    "ActivationSource",
    "FnSource",
    "ProbeScoreSource",
    "QwenEncoderSource",
    "RuntimeProbeScoreSource",
    "LiveProbeScoreSource",
    "open_store",
    "build_structured_activations",
    "doc_hash",
    "hash_seeded_noise",
    "make_orthonormal_P",
]
