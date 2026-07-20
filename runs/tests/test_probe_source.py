"""ConceptProbeScoreSource tests: the present_z realism threshold zeroes exactly the
non-firing rows, and the family/column-order guard fires. No network, no tokenizer."""
import numpy as np
import pytest

from nanochat.injection.sources import RuntimeProbeScoreSource, _RuntimeProbeBase

import concepts
from probe_source import ConceptProbeScoreSource


def _threshold_only(present_z, aligned, monkeypatch):
    """Exercise _align_and_gather's threshold with the parent's alignment stubbed."""
    src = ConceptProbeScoreSource.__new__(ConceptProbeScoreSource)
    src.present_z = float(present_z)
    monkeypatch.setattr(_RuntimeProbeBase, "_align_and_gather",
                        lambda self, text, n_tokens, z: aligned)
    return src._align_and_gather("text", aligned.shape[0], None)


def test_rows_below_threshold_are_exactly_zero(monkeypatch):
    a = np.array([[2.5, 0.1, -1.0, 0.0],      # max 2.5 -> keep
                  [1.9, 1.8, 0.0, 0.0],       # max 1.9 -> zero
                  [-5.0, -3.0, -1.0, -0.5],   # all negative -> zero
                  [0.0, 0.0, 0.0, 2.0]], np.float32)   # max == 2.0 -> keep (>=)
    out = _threshold_only(2.0, a.copy(), monkeypatch)
    assert np.array_equal(out[0], a[0])
    assert np.array_equal(out[3], a[3])
    assert not out[1].any() and not out[2].any()
    assert (out[1] == 0.0).all() and (out[2] == 0.0).all()   # exact zero, not small


def test_threshold_is_configurable(monkeypatch):
    a = np.array([[1.5, 0.0], [0.4, 0.0]], np.float32)
    assert _threshold_only(1.0, a.copy(), monkeypatch)[0].any()
    assert not _threshold_only(1.0, a.copy(), monkeypatch)[1].any()
    assert not _threshold_only(3.0, a.copy(), monkeypatch)[0].any()


def test_none_passthrough(monkeypatch):
    src = ConceptProbeScoreSource.__new__(ConceptProbeScoreSource)
    src.present_z = 2.0
    monkeypatch.setattr(_RuntimeProbeBase, "_align_and_gather",
                        lambda self, text, n_tokens, z: None)
    assert src._align_and_gather("t", 3, None) is None


def _construct(monkeypatch, columns, **kw):
    """Real ConceptProbeScoreSource.__init__ with the store-opening parent stubbed."""
    def fake_init(self, *args, concepts=None, **kwargs):
        self.concepts = list(concepts)
        self.r = len(self.concepts)
    monkeypatch.setattr(RuntimeProbeScoreSource, "__init__", fake_init)
    return ConceptProbeScoreSource("store", [0], concepts=columns, **kw)


@pytest.mark.parametrize("family", ["seasons", "weekdays"])
def test_family_guard_accepts_store_order(monkeypatch, family):
    fam = concepts.get_family(family)
    src = _construct(monkeypatch, list(fam.store_order), family=family, present_z=2.0)
    assert src.present_z == 2.0
    assert src.family == family
    assert list(src.concepts) == list(fam.store_order)


@pytest.mark.parametrize("family", ["seasons", "weekdays"])
def test_family_guard_rejects_cycle_order_columns(monkeypatch, family):
    fam = concepts.get_family(family)
    with pytest.raises(ValueError, match="store order"):
        _construct(monkeypatch, list(fam.cycle_order), family=family)


def test_family_is_optional(monkeypatch):
    # present_z defaults OFF: the site's relu owns thresholding under dose.
    src = _construct(monkeypatch, ["autumn", "spring"])
    assert src.family is None and src.present_z == 0.0


def test_class_name_matches_the_config_hook_assertion():
    # injection_train asserts type(src).__name__ == the config 'class' suffix, and
    # that the class subclasses RuntimeProbeScoreSource.
    assert ConceptProbeScoreSource.__name__ == "ConceptProbeScoreSource"
    assert issubclass(ConceptProbeScoreSource, RuntimeProbeScoreSource)
