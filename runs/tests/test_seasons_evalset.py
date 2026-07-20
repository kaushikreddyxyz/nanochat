"""CPU tests for seasons_evalset: byte-determinism, committed-jsonl freshness, schema
conformance, cycle arithmetic (incl. k>1 wraparound), month<->season table consistency
and hemisphere tagging, polysemy labeling, and weekday-content isolation. No torch.
Run: python -m pytest runs/tests/test_seasons_evalset.py
"""
import os
import re
from collections import Counter, defaultdict

import seasons_evalset as S
from conftest import SEASONS
from seasons_concepts import CYCLE_ORDER, R, season_plus

HERE = os.path.join(SEASONS, "eval")

WEEKDAY_WORDS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
                 "sunday", "weekday", "weekend"]


def _items():
    return S.generate_items()


# --------------------------------------------------------------------------- #
# Generation: determinism, freshness, schema.
# --------------------------------------------------------------------------- #
def test_deterministic_bytes():
    assert S.to_jsonl_bytes(_items()) == S.to_jsonl_bytes(_items()), \
        "generation is not byte-deterministic"


def test_committed_jsonl_is_fresh():
    path = os.path.join(HERE, "evalsets", "seasons_v1.jsonl")
    assert os.path.exists(path), f"missing generated eval set: {path}"
    with open(path, "rb") as f:
        on_disk = f.read()
    assert on_disk == S.to_jsonl_bytes(_items()), \
        "committed seasons_v1.jsonl is stale — re-run seasons_evalset.py"


def test_unique_ids_and_size():
    items = _items()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    assert 300 <= len(items) <= 500, f"item count {len(items)} outside 300-500"


def test_schema_conformance():
    items = _items()
    for it in items:
        assert set(it) == {"id", "category", "prompt", "options", "answer", "meta"}, it["id"]
        assert it["category"] in S.CATEGORIES, (it["id"], it["category"])
        assert isinstance(it["prompt"], str) and it["prompt"], it["id"]
        assert not it["prompt"].endswith(" "), (it["id"], "prompt must not end with a space")
        assert it["answer"] in it["options"], (it["id"], "answer not in options")
        assert len(it["options"]) == len(set(it["options"])), (it["id"], "duplicate options")
        assert len(it["options"]) >= 2, (it["id"], "need >=2 options")
        for key in ("template", "fill_in", "hemisphere", "hemisphere_stated",
                    "expect_season_concept", "answer_kind"):
            assert key in it["meta"], (it["id"], f"missing meta.{key}")
        assert it["meta"]["answer_kind"] == S.answer_kind(it["answer"]), it["id"]
        if it["meta"]["expect_season_concept"]:
            assert it["meta"]["answer_kind"] == "season", (it["id"], "season answer expected")


def test_option_sets_match_answer_kind():
    for it in _items():
        if it["category"] in ("order_next", "order_prev", "order_k", "fact_position",
                              "characteristic", "usage_context", "solstice_equinox",
                              "month_to_season"):
            assert it["options"] == S.SEASONS, (it["id"], "expected the 4 seasons as options")
        if it["meta"]["answer_kind"] == "month" and it["category"] != "polysemy":
            assert it["options"] == S.MONTHS, (it["id"], "month answers use the 12 months")


def test_no_answer_leak_in_fill_in():
    leaks = [it["id"] for it in _items() if it["meta"]["fill_in"] and S.leaks_answer(it)]
    assert not leaks, f"fill-in items leak the answer into the prompt: {leaks}"


def test_categories_have_multiple_templates():
    tmpls = defaultdict(set)
    for it in _items():
        tmpls[it["category"]].add(it["meta"]["template"])
    for cat in ("order_next", "order_prev", "order_k", "fact_position",
                "month_to_season", "usage_context"):
        assert len(tmpls[cat]) >= 3, f"{cat} has <3 surface templates ({len(tmpls[cat])})"


def test_balanced_season_counts():
    c = Counter(it["answer"] for it in _items()
                if it["category"] in S.BALANCED_SEASON_CATEGORIES)
    assert set(c) == set(CYCLE_ORDER), f"balanced categories miss seasons: {set(c)}"
    assert len(set(c.values())) == 1, f"per-season answer counts not balanced: {dict(c)}"


# --------------------------------------------------------------------------- #
# Cycle arithmetic, including k>1 wraparound.
# --------------------------------------------------------------------------- #
def test_season_plus_wraps():
    assert [season_plus(s, 1) for s in CYCLE_ORDER] == ["summer", "autumn", "winter", "spring"]
    assert [season_plus(s, -1) for s in CYCLE_ORDER] == ["winter", "spring", "summer", "autumn"]
    for s in CYCLE_ORDER:
        assert season_plus(s, R) == s, "full turn must be the identity"
        assert season_plus(s, 2) == season_plus(s, -2), "k=2 is the antipode on a 4-cycle"
        assert season_plus(s, 3) == season_plus(s, -1)
        assert season_plus(season_plus(s, 2), 2) == s


def test_order_items_match_cycle_arithmetic():
    seen_k = set()
    for it in _items():
        m = it["meta"]
        if it["category"] == "order_next":
            assert it["answer"] == season_plus(m["season"], 1), it["id"]
        elif it["category"] == "order_prev":
            assert it["answer"] == season_plus(m["season"], -1), it["id"]
        elif it["category"] == "order_k":
            k = m["k"] if m["direction"] == "after" else -m["k"]
            assert it["answer"] == season_plus(m["season"], k), it["id"]
            seen_k.add((m["k"], m["direction"]))
    assert seen_k == {(2, "after"), (2, "before"), (3, "after"), (3, "before")}, \
        f"order_k must sweep k>1 both ways, got {sorted(seen_k)}"


def test_fact_position_matches_stated_convention():
    for it in _items():
        if it["category"] != "fact_position":
            continue
        m = it["meta"]
        assert it["answer"] == season_plus(m["convention"], m["position"] - 1), it["id"]
        assert m["position"] >= 2, (it["id"], "position 1 would leak the start season")
        assert m["convention"] in it["prompt"], (it["id"], "convention must be stated")


# --------------------------------------------------------------------------- #
# Month tables + hemisphere handling.
# --------------------------------------------------------------------------- #
def test_month_tables_self_consistent():
    for hemi in ("northern", "southern"):
        months = [m for ms in S.SEASON_MONTHS[hemi].values() for m in ms]
        assert sorted(months) == sorted(S.MONTHS), f"{hemi} months are not a partition"
        for season, ms in S.SEASON_MONTHS[hemi].items():
            assert len(ms) == 3, (hemi, season)
            for m in ms:
                assert S.MONTH_SEASON[hemi][m] == season, (hemi, m)
    # the hemispheres are exact antipodes: every month is opposite across the equator
    for m in S.MONTHS:
        north = S.MONTH_SEASON["northern"][m]
        assert S.MONTH_SEASON["southern"][m] == season_plus(north, 2), m


def test_hemisphere_tagging():
    for it in _items():
        m = it["meta"]
        assert m["hemisphere"] in (None, "northern", "southern"), it["id"]
        if m["hemisphere"] is None:
            assert m["hemisphere_stated"] is False, it["id"]
            continue
        if it["category"] == "month_to_season":
            assert it["answer"] == S.MONTH_SEASON[m["hemisphere"]][m["month"]], it["id"]
        # a southern item is unanswerable unless the prompt names its hemisphere
        if m["hemisphere"] == "southern":
            assert m["hemisphere_stated"], (it["id"], "southern items must state it")
            assert re.search(r"southern|Australia", it["prompt"]), it["id"]
        if m["hemisphere_stated"] and m["hemisphere"] == "northern":
            assert "northern" in it["prompt"], it["id"]


def test_both_hemispheres_are_covered():
    c = Counter(it["meta"]["hemisphere"] for it in _items())
    assert c["northern"] > 0 and c["southern"] > 0, f"hemisphere coverage: {dict(c)}"
    unstated = [it["id"] for it in _items()
                if it["meta"]["hemisphere"] == "northern" and not it["meta"]["hemisphere_stated"]]
    assert unstated, "keep an unstated-northern-default slice to measure the default"


def test_season_to_month_matches_the_table():
    part_index = {"first": 0, "middle": 1, "last": 2}
    for it in _items():
        if it["category"] != "season_to_month":
            continue
        m = it["meta"]
        months = S.SEASON_MONTHS[m["hemisphere"]][m["season"]]
        assert it["answer"] == months[part_index[m["part"]]], it["id"]
        assert m["expect_season_concept"] is False, (it["id"], "answer is a month")


# --------------------------------------------------------------------------- #
# Controls: polysemy + sanity + weekday isolation.
# --------------------------------------------------------------------------- #
def test_polysemy_items_are_labeled_and_non_seasonal():
    poly = [it for it in _items() if it["category"] == "polysemy"]
    assert len(poly) >= 15, f"only {len(poly)} polysemy controls"
    kinds = Counter()
    for it in poly:
        m = it["meta"]
        assert m["expect_season_concept"] is False, \
            (it["id"], "the correct answer must not denote a season")
        assert m["kind"] in ("non_season_sense", "season_word_answer"), it["id"]
        assert m["season_word"] in S.SEASONS + ["fall"], it["id"]
        assert isinstance(m["sense"], str) and m["sense"], it["id"]
        # the season word must actually be present in the prompt or be the answer
        assert (S.leaks_word(it["prompt"], m["season_word"])
                or it["answer"] == m["season_word"]), (it["id"], "no season word to fire on")
        kinds[m["kind"]] += 1
    assert kinds["non_season_sense"] >= 5 and kinds["season_word_answer"] >= 5, dict(kinds)
    # the non-season senses are what the weekday suite had no analogue for
    senses = {it["meta"]["sense"] for it in poly if it["meta"]["kind"] == "non_season_sense"}
    assert {"water_source", "coil", "jump", "drop"} <= senses, senses


def test_sanity_items_are_season_free():
    for it in _items():
        if it["category"] != "sanity":
            continue
        assert it["meta"]["expect_season_concept"] is False, it["id"]
        blob = " ".join([it["prompt"], it["answer"]] + it["options"])
        for w in S.SEASONS + ["fall"]:
            assert not S.leaks_word(blob, w), (it["id"], f"sanity item mentions {w!r}")


def test_no_weekday_content_leaks_in():
    for it in _items():
        blob = " ".join([it["prompt"], it["answer"]] + it["options"])
        for w in WEEKDAY_WORDS:
            assert not S.leaks_word(blob, w), (it["id"], f"weekday content leaked: {w!r}")


def test_fall_surface_form_items():
    fall = [it for it in _items() if it["category"] == "synonym_fall"]
    assert fall, "no 'fall' surface-form items"
    for it in fall:
        assert it["meta"]["surface_form"] == "fall" and it["meta"]["canonical"] == "autumn"
        assert S.leaks_word(it["prompt"], "fall"), it["id"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
