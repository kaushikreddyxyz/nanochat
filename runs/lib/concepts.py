"""Concept-family registry: probe-store column order/indices vs canonical cycle
order, and the explicit map between them. Import: sys.path.insert this dir, then
``import concepts``.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptFamily:
    """One family's two orderings.

    ``store_order`` is the probe-score store's own column order (families grouped,
    NAME-SORTED within a family) and is what every injection channel index means.
    ``cycle_order`` is the semantic cycle used for geometry (phase angles). The two
    are DIFFERENT permutations; mixing them silently attaches phases to the wrong
    channels, so every accessor here names which order it speaks.
    """

    name: str
    store_order: tuple
    store_index: tuple
    cycle_order: tuple = None    # None where no uniform cycle is pinned

    @property
    def r(self):
        return len(self.store_order)

    @property
    def is_cyclic(self):
        return self.cycle_order is not None

    def store_row(self, concept):
        return self.store_order.index(concept)

    def store_column(self, concept):
        return self.store_index[self.store_row(concept)]

    def cycle_position(self, concept):
        assert self.is_cyclic, f"family {self.name!r} has no pinned cycle order"
        return self.cycle_order.index(concept)

    def cycle_position_of_store_row(self):
        """[r] cycle position for each STORE row — the store->cycle map."""
        return [self.cycle_position(c) for c in self.store_order]

    def store_row_of_cycle_position(self):
        """[r] store row for each CYCLE position — the cycle->store map."""
        return [self.store_row(c) for c in self.cycle_order]


FAMILIES = {
    f.name: f
    for f in (
        ConceptFamily(
            name="color_wheel",
            store_order=("blue", "blue-green", "green", "orange", "red",
                         "red-orange", "violet", "yellow", "yellow-green"),
            store_index=tuple(range(0, 9)),
            # the 9 stored hues are not uniformly spaced on the wheel; no cycle pinned
        ),
        ConceptFamily(
            name="continents",
            store_order=("africa", "asia", "europe", "north_america", "oceania",
                         "south_america"),
            store_index=tuple(range(9, 15)),
        ),
        ConceptFamily(
            name="directions",
            store_order=("east", "north", "northeast", "northwest", "south",
                         "southeast", "southwest", "west"),
            store_index=tuple(range(15, 23)),
            cycle_order=("north", "northeast", "east", "southeast", "south",
                         "southwest", "west", "northwest"),
        ),
        ConceptFamily(
            name="months",
            store_order=("april", "august", "december", "february", "january",
                         "july", "june", "march", "may", "november", "october",
                         "september"),
            store_index=tuple(range(23, 35)),
            cycle_order=("january", "february", "march", "april", "may", "june",
                         "july", "august", "september", "october", "november",
                         "december"),
        ),
        ConceptFamily(
            name="moon_phases",
            store_order=("first_quarter", "full_moon", "last_quarter", "new_moon",
                         "waning_crescent", "waning_gibbous", "waxing_crescent",
                         "waxing_gibbous"),
            store_index=tuple(range(35, 43)),
            cycle_order=("new_moon", "waxing_crescent", "first_quarter",
                         "waxing_gibbous", "full_moon", "waning_gibbous",
                         "last_quarter", "waning_crescent"),
        ),
        ConceptFamily(
            name="seasons",
            store_order=("autumn", "spring", "summer", "winter"),
            store_index=(43, 44, 45, 46),
            cycle_order=("spring", "summer", "autumn", "winter"),
        ),
        ConceptFamily(
            name="weekdays",
            store_order=("friday", "monday", "saturday", "sunday", "thursday",
                         "tuesday", "wednesday"),
            store_index=(47, 48, 49, 50, 51, 52, 53),
            cycle_order=("monday", "tuesday", "wednesday", "thursday", "friday",
                         "saturday", "sunday"),
        ),
    )
}


def get_family(name):
    if name not in FAMILIES:
        raise KeyError(f"unknown concept family {name!r}; registered: {sorted(FAMILIES)}")
    return FAMILIES[name]


def assert_matches_store_columns(store_concepts):
    """HARD: every family's pinned (name, column) pairs must equal the real store
    column list (probe_set.json 'main_block_concepts' / columns.json 'concepts')."""
    for fam in FAMILIES.values():
        for name, col in zip(fam.store_order, fam.store_index):
            got = store_concepts[col]
            if got != name:
                raise AssertionError(
                    f"{fam.name}: store column {col} is {got!r}, registry says {name!r}")
            if store_concepts.index(name) != col:
                raise AssertionError(
                    f"{fam.name}: {name!r} first appears at column "
                    f"{store_concepts.index(name)}, registry pins {col}")


def _self_check():
    seen = {}
    for fam in FAMILIES.values():
        assert len(fam.store_order) == len(fam.store_index), fam.name
        assert list(fam.store_order) == sorted(fam.store_order), \
            f"{fam.name}: store_order must be name-sorted"
        assert list(fam.store_index) == list(range(fam.store_index[0],
                                                   fam.store_index[0] + fam.r)), \
            f"{fam.name}: store columns must be contiguous ascending"
        for name, col in zip(fam.store_order, fam.store_index):
            assert col not in seen, f"column {col} claimed by {seen.get(col)} and {fam.name}"
            seen[col] = fam.name
        if fam.is_cyclic:
            assert sorted(fam.cycle_order) == sorted(fam.store_order), \
                f"{fam.name}: cycle_order is not a permutation of store_order"
            assert fam.r >= 3, f"{fam.name}: a cycle needs >= 3 points"
    assert sorted(seen) == list(range(len(seen))), "store columns are not a 0..n-1 cover"


_self_check()
