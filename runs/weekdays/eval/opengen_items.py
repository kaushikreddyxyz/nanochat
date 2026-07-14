"""Deterministic DAY-FREE prompt set for the weekday-injection OPEN-GENERATION
battery (the follow-up to the causal/counterfactual eval).

The scientific point (user's words): "provide no information about Mark but do
something like 'The day Mark was born on is {}' ... and inject at every given
position." Every prompt here is completion-style (no chat formatting — 124M base
models), short, and ENDS exactly where a weekday word would naturally continue.
Crucially NONE of them contains a weekday anywhere: the clean activations are
therefore EXACTLY zero, so any day that appears in the continuation is either the
model's own natural propensity (the `none` condition) or a product of the
injected day pattern — never a copy of a day word that was in the prompt.

Three families (the user's three shapes), ~30 prompts total:
  * ``personal_fact``  — "The day Mark was born on is", a stated-fact framing whose
    completion is a specific weekday the model has NO way to know.
  * ``forward_looking`` — "My birthday is next", a future-schedule framing.
  * ``schedule``        — "My {item} delivers on", a delivery/appointment framing;
    one base template swept over ~6 concrete item nouns plus a few variants.

Pure module: stdlib only, no torch / no harness. ``generate_items()`` is a
deterministic function of the literals below.
"""
from __future__ import annotations

# The 7 weekday names (lowercase), for the generation-side day detector. Kept
# local (this module is day-FREE by construction, so it needs the names only to
# describe what a completion might contain, never to build a prompt).
DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday"]


# --------------------------------------------------------------------------- #
# Family 1 — personal fact. A stated day the model cannot possibly know; the
# completion is whatever weekday the model's prior (or the injection) supplies.
# Each ends right before the weekday word (mostly at the article "a" or a verb).
# --------------------------------------------------------------------------- #
_PERSONAL_FACT = [
    ("PF1", "The day Mark was born on is"),
    ("PF2", "Sarah always says her lucky day is"),
    ("PF3", "The day David got married was a"),
    ("PF4", "Elena told everyone the accident happened on a"),
    ("PF5", "James clearly remembers being hired on a"),
    ("PF6", "The day the twins arrived was a"),
    ("PF7", "Grandpa's favorite day of the week has always been"),
    ("PF8", "According to the certificate, the birth took place on a"),
    ("PF9", "The day I first met her was a"),
    ("PF10", "He was baptized on a"),
]

# --------------------------------------------------------------------------- #
# Family 2 — forward-looking. A future event with an as-yet-unstated weekday.
# --------------------------------------------------------------------------- #
_FORWARD_LOOKING = [
    ("FL1", "My birthday is next"),
    ("FL2", "The wedding is happening this coming"),
    ("FL3", "Our flight leaves next"),
    ("FL4", "The big meeting is scheduled for this"),
    ("FL5", "I get my results back next"),
    ("FL6", "The concert is on"),
    ("FL7", "We are moving out next"),
    ("FL8", "The final exam takes place this"),
    ("FL9", "Her baby is due next"),
    ("FL10", "The product launch is planned for next"),
]

# --------------------------------------------------------------------------- #
# Family 3 — schedule / delivery. One base template swept over 6 concrete item
# nouns ("My {item} delivers on") + 4 appointment/renewal variants.
# --------------------------------------------------------------------------- #
_SCHEDULE_ITEMS = ["package", "sofa", "laptop", "mattress", "printer", "refrigerator"]
_SCHEDULE_VARIANTS = [
    ("SD2", "The plumber is coming on"),
    ("SD3", "The delivery is scheduled for"),
    ("SD4", "My subscription renews every"),
    ("SD5", "The garbage is collected on"),
]


def generate_items():
    """Deterministic list of item dicts. Fields:
      id, family ('personal_fact'|'forward_looking'|'schedule'),
      template ('PF1'.. / 'FL1'.. / 'SD1'/'SD2'..), item (schedule noun or None),
      prompt (day-FREE, completion-style).
    """
    items = []
    for tid, prompt in _PERSONAL_FACT:
        items.append({"id": f"personal_fact/{tid}", "family": "personal_fact",
                      "template": tid, "item": None, "prompt": prompt})
    for tid, prompt in _FORWARD_LOOKING:
        items.append({"id": f"forward_looking/{tid}", "family": "forward_looking",
                      "template": tid, "item": None, "prompt": prompt})
    for noun in _SCHEDULE_ITEMS:  # the {item} sweep, one shared base template SD1
        items.append({"id": f"schedule/SD1/{noun}", "family": "schedule",
                      "template": "SD1", "item": noun,
                      "prompt": f"My {noun} delivers on"})
    for tid, prompt in _SCHEDULE_VARIANTS:
        items.append({"id": f"schedule/{tid}", "family": "schedule",
                      "template": tid, "item": None, "prompt": prompt})
    return items


def families():
    """Ordered family names (for summary tables)."""
    return ["personal_fact", "forward_looking", "schedule"]


if __name__ == "__main__":
    its = generate_items()
    from collections import Counter
    c = Counter(i["family"] for i in its)
    print(f"{len(its)} day-free prompts: " + ", ".join(f"{k}={c[k]}" for k in families()))
    # No prompt may contain a weekday — that is the whole point of this set.
    for it in its:
        low = it["prompt"].lower()
        assert not any(d in low for d in DAY_NAMES), f"prompt mentions a weekday: {it['id']}"
    for it in its[:3] + its[10:12] + its[20:23]:
        print(" ", it["id"], "->", repr(it["prompt"]))
