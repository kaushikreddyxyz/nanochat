"""ConceptProbeScoreSource: RuntimeProbeScoreSource for one concept family plus a
POST-alignment realism threshold. Wired from an --activation-config source spec via
"class": "runs/lib/probe_source.py:ConceptProbeScoreSource".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanochat.injection.sources import RuntimeProbeScoreSource  # noqa: E402

import concepts as concept_registry  # noqa: E402


class ConceptProbeScoreSource(RuntimeProbeScoreSource):
    """Parent handles column subsetting (pass ``concepts=`` in STORE order); this
    subclass only adds:
      present_z : float (default 0.0, i.e. off — the SITE's relu thresholds) — a
                  nanochat-token row survives only if its max channel z >= present_z,
                  else the row is zeroed.
      family    : str (optional) — registry family name; when given, the spec's
                  ``concepts`` list must EQUAL that family's store order exactly.
    """

    def __init__(self, *args, present_z=0.0, family=None, **kwargs):
        self.present_z = float(present_z)
        self.family = family
        super().__init__(*args, **kwargs)
        if family is not None:
            fam = concept_registry.get_family(family)
            if list(self.concepts) != list(fam.store_order):
                raise ValueError(
                    f"family {family!r}: source columns {list(self.concepts)} != registry "
                    f"store order {list(fam.store_order)}. Refusing to start — a permutation "
                    f"here silently relabels every injection channel.")

    def _align_and_gather(self, text, n_tokens, z_gemma):
        out = super()._align_and_gather(text, n_tokens, z_gemma)
        if out is None:
            return None
        # Exact zero (not small) — a zero row is the site's strict injection no-op.
        keep = out.max(axis=1) >= self.present_z
        out[~keep] = 0.0
        return out
