"""Registry tests: store column/name consistency (against the real probe_set.json
when present) and the store<->cycle order mapping."""
import json

import pytest

import concepts
from manifold import find_attribution_out

WEEKDAY_STORE = ["friday", "monday", "saturday", "sunday", "thursday", "tuesday",
                 "wednesday"]
SEASON_STORE = ["autumn", "spring", "summer", "winter"]


def test_registered_families_cover_the_54_store_columns():
    cols = sorted(c for f in concepts.FAMILIES.values() for c in f.store_index)
    assert cols == list(range(54))


@pytest.mark.parametrize("family,store,index", [
    ("weekdays", WEEKDAY_STORE, list(range(47, 54))),
    ("seasons", SEASON_STORE, [43, 44, 45, 46]),
])
def test_pinned_store_order_and_columns(family, store, index):
    fam = concepts.get_family(family)
    assert list(fam.store_order) == store
    assert list(fam.store_index) == index
    assert fam.r == len(store)


def test_store_and_cycle_orders_differ_and_map_consistently():
    for fam in concepts.FAMILIES.values():
        if not fam.is_cyclic:
            continue
        to_cycle = fam.cycle_position_of_store_row()
        to_store = fam.store_row_of_cycle_position()
        assert sorted(to_cycle) == list(range(fam.r))
        assert sorted(to_store) == list(range(fam.r))
        # the two maps are inverses
        for row in range(fam.r):
            assert to_store[to_cycle[row]] == row
        for name in fam.store_order:
            assert fam.store_order[fam.store_row(name)] == name
            assert fam.cycle_order[fam.cycle_position(name)] == name


def test_weekday_and_season_cycle_orders_are_not_the_store_orders():
    wk = concepts.get_family("weekdays")
    se = concepts.get_family("seasons")
    assert list(wk.cycle_order) != list(wk.store_order)
    assert list(se.cycle_order) != list(se.store_order)
    # the load-bearing specifics: monday is store row 1 but cycle position 0
    assert wk.store_row("monday") == 1 and wk.cycle_position("monday") == 0
    assert se.store_row("spring") == 1 and se.cycle_position("spring") == 0
    assert se.cycle_position("autumn") == 2 and se.store_row("autumn") == 0


def test_non_cyclic_families_refuse_cycle_queries():
    fam = concepts.get_family("continents")
    assert not fam.is_cyclic
    with pytest.raises(AssertionError):
        fam.cycle_position("europe")


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        concepts.get_family("fortnights")


def test_registry_matches_real_probe_set_columns():
    try:
        attr_out = find_attribution_out(__file__)
    except FileNotFoundError:
        pytest.skip("attribution/out not available")
    meta = json.load(open(attr_out / "probe_set.json"))
    concepts.assert_matches_store_columns(list(meta["main_block_concepts"]))


def test_column_mismatch_is_detected():
    bad = ["x"] * 54
    with pytest.raises(AssertionError):
        concepts.assert_matches_store_columns(bad)
