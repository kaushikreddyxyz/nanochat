"""CPU tests for the seasons causal/open-generation item sets and the registry binding:
store-vs-cycle order separation, counterfactual answer arithmetic, implant prompts being
season-free, and open-gen prompts naming no season/month/weekday. No torch.
Run: python -m pytest runs/tests/test_seasons_items.py
"""
import seasons_concepts as SC
import seasons_evalset as S
import seasons_items as ci
import seasons_opengen_items as oi
from test_seasons_evalset import WEEKDAY_WORDS


# --------------------------------------------------------------------------- #
# Registry binding — the eval suite must not carry its own ordering data.
# --------------------------------------------------------------------------- #
def test_binding_matches_shared_registry():
    fam = SC.FAMILY
    assert fam.name == "seasons"
    assert SC.STORE_ORDER == list(fam.store_order) == ["autumn", "spring", "summer", "winter"]
    assert SC.STORE_COLUMNS == list(fam.store_index) == [43, 44, 45, 46]
    assert SC.CYCLE_ORDER == list(fam.cycle_order) == ["spring", "summer", "autumn", "winter"]
    assert SC.R == fam.r == 4


def test_store_and_cycle_orders_are_distinct():
    assert SC.STORE_ORDER != SC.CYCLE_ORDER, "store order must not be the cycle order"
    assert sorted(SC.STORE_ORDER) == sorted(SC.CYCLE_ORDER)
    # a channel index and a cycle position are different numbers for most seasons
    differing = [s for s in SC.CYCLE_ORDER if ci.store_idx(s) != ci.cycle_idx(s)]
    assert len(differing) >= 3, f"store/cycle indices barely differ: {differing}"
    assert ci.store_idx("autumn") == 0 and ci.cycle_idx("autumn") == 2


# --------------------------------------------------------------------------- #
# seasons_items
# --------------------------------------------------------------------------- #
def test_causal_ids_unique_and_counts():
    items = ci.generate_items()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate causal item ids"
    n_mention = sum(1 for i in items if i["family"] == "mention")
    n_implant = sum(1 for i in items if i["family"] == "implant")
    assert n_mention == len(ci.NAMES) * len(ci._MENTION_CONTEXTS) * SC.R
    assert n_implant == len(ci.NAMES) * 2 * SC.R
    assert len(items) == n_mention + n_implant


def test_mention_items_carry_cycle_arithmetic():
    for it in ci.generate_items():
        if it["family"] != "mention":
            continue
        assert it["text_answer"] == SC.season_plus(it["season"], it["kq"]), it["id"]
        assert S.leaks_word(it["prompt"], it["season"]), (it["id"], "X must be named in text")
        assert not it["prompt"].endswith(" "), it["id"]
    kqs = {it["kq"] for it in ci.generate_items() if it["family"] == "mention"}
    assert kqs == {0, 1, -1, 2}, f"expected recall + both neighbours + antipode, got {kqs}"


def test_implant_prompts_are_season_free():
    for it in ci.generate_items():
        if it["family"] != "implant":
            continue
        for w in SC.CYCLE_ORDER + ["fall"]:
            assert not S.leaks_word(it["prompt"], w), (it["id"], f"implant names {w!r}")
        assert it["text_answer"] is None, it["id"]
        cs, ce = it["noun_char"]
        assert it["prompt"][cs:ce].strip() == it["noun"], it["id"]


def test_noun_token_span_on_a_whitespace_tokenizer():
    def encode(text):
        return text.split()

    prompt = "Mark was born in a small town. Asked which season, the answer is"
    cs = prompt.index(" town")
    start, end = ci.noun_token_span(encode, prompt, cs, cs + len(" town"))
    assert (start, end) == (6, 7), (start, end)
    assert encode(prompt)[start:end] == ["town."]


def test_completions_are_lowercase_and_spaced():
    for s in SC.CYCLE_ORDER:
        assert ci.completion(s) == " " + s
        assert s.islower()


# --------------------------------------------------------------------------- #
# seasons_opengen_items
# --------------------------------------------------------------------------- #
def test_opengen_prompts_are_season_month_and_weekday_free():
    items = oi.generate_items()
    assert len(items) == 30, len(items)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate opengen ids"
    for it in items:
        assert it["family"] in oi.families(), it["id"]
        assert not it["prompt"].endswith(" "), it["id"]
        for w in oi.SEASON_SURFACE_FORMS + WEEKDAY_WORDS + [m.lower() for m in S.MONTHS]:
            assert not S.leaks_word(it["prompt"], w), (it["id"], f"prompt names {w!r}")


def test_opengen_families_are_balanced():
    from collections import Counter
    c = Counter(it["family"] for it in oi.generate_items())
    assert set(c) == set(oi.families())
    assert len(set(c.values())) == 1, f"family sizes differ: {dict(c)}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
