"""Oracle-feature injection for nanochat.

Sibling of ``modular_addition/oracle``: a frozen additive feature ("oracle")
that is a pure function of the token id, injected into the residual stream right
before the transformer trunk (see the hook in ``nanochat.gpt.GPT.forward``). The
first feature family is *geometric manifolds* — token ids placed on a ring,
line, sphere, or helix in a few reserved residual dimensions.

Public API:
    attach_oracle, detach_oracle, freeze_params
    make_table_oracle, make_manifold_oracle, make_ring_oracle
    ring_coords, line_coords, sphere_coords, helix_coords
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
]
