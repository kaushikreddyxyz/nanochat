"""Deterministic SEASON-FREE prompt set for open generation: 30 completion-style prompts
(personal_fact / forward_looking / seasonal_activity families) that end where a season
would continue but name no season, no month and no weekday — so clean acts are exactly
zero and any generated season is prior or injection, never a copied prompt word.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seasons_concepts import CYCLE_ORDER  # noqa: E402

# Surface forms the generation-side detector must look for: the four season names plus
# the American synonym for autumn. This module contains NONE of them by construction.
SEASON_SURFACE_FORMS = list(CYCLE_ORDER) + ["fall"]

# Interface the shared open-gen driver (runs/lib/eval/open_gen.py, via
# --opengen-module) reads: the surface forms to detect, and each -> its CANONICAL
# label (only 'fall' is a synonym; the four season names map to themselves).
SURFACE_FORMS = SEASON_SURFACE_FORMS
SURFACE_TO_CANONICAL = {"fall": "autumn"}

# Family 1 — personal fact. A season the model cannot possibly know; the completion is
# whatever season the prior (or the injection) supplies.
_PERSONAL_FACT = [
    ("PF1", "Mark's favorite time of the year has always been"),
    ("PF2", "The season Sarah was born in is"),
    ("PF3", "David says the happiest part of the year for him is"),
    ("PF4", "Elena told everyone the accident happened in the"),
    ("PF5", "James clearly remembers being hired in the"),
    ("PF6", "The twins arrived in the"),
    ("PF7", "Grandpa's favorite season has always been"),
    ("PF8", "According to the certificate, the birth took place in the"),
    ("PF9", "The first time I met her it was"),
    ("PF10", "He was baptized in the"),
]

# Family 2 — forward-looking. A future event with an as-yet-unstated season.
_FORWARD_LOOKING = [
    ("FL1", "My wedding is planned for next"),
    ("FL2", "The family reunion is happening this coming"),
    ("FL3", "Our long trip abroad begins next"),
    ("FL4", "The new building should be finished by next"),
    ("FL5", "I get my exam results back in the"),
    ("FL6", "The festival takes place every"),
    ("FL7", "We are moving to the coast next"),
    ("FL8", "Her baby is due next"),
    ("FL9", "The product launch is planned for next"),
    ("FL10", "The team's next tournament is in the"),
]

# Family 3 — seasonal activity. One base template swept over 6 concrete objects
# ("The best time of year to use my {object} is") plus 4 event variants.
_ACTIVITY_OBJECTS = ["skis", "surfboard", "raincoat", "sunhat", "sledge", "umbrella"]
_ACTIVITY_VARIANTS = [
    ("SA2", "The farmers gather their main crop in the"),
    ("SA3", "The lake usually freezes over in the"),
    ("SA4", "The tourists arrive in the largest numbers in the"),
    ("SA5", "The trees are at their greenest in the"),
]


def generate_items():
    """Deterministic list of item dicts. Fields: id, family, template, object
    (activity object or None), prompt (season-free, completion-style)."""
    items = []
    for tid, prompt in _PERSONAL_FACT:
        items.append({"id": f"personal_fact/{tid}", "family": "personal_fact",
                      "template": tid, "object": None, "prompt": prompt})
    for tid, prompt in _FORWARD_LOOKING:
        items.append({"id": f"forward_looking/{tid}", "family": "forward_looking",
                      "template": tid, "object": None, "prompt": prompt})
    for obj in _ACTIVITY_OBJECTS:
        items.append({"id": f"seasonal_activity/SA1/{obj}", "family": "seasonal_activity",
                      "template": "SA1", "object": obj,
                      "prompt": f"The best time of year to use my {obj} is"})
    for tid, prompt in _ACTIVITY_VARIANTS:
        items.append({"id": f"seasonal_activity/{tid}", "family": "seasonal_activity",
                      "template": tid, "object": None, "prompt": prompt})
    return items


def families():
    """Ordered family names (for summary tables)."""
    return ["personal_fact", "forward_looking", "seasonal_activity"]


if __name__ == "__main__":
    from collections import Counter
    its = generate_items()
    c = Counter(i["family"] for i in its)
    print(f"{len(its)} season-free prompts: " + ", ".join(f"{k}={c[k]}" for k in families()))
    for it in its[:3] + its[10:12] + its[20:23]:
        print(" ", it["id"], "->", repr(it["prompt"]))
