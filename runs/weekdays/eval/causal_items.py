"""Deterministic item set for the weekday-geometry CAUSAL / COUNTERFACTUAL eval.

Two item families, both completion-style (no chat formatting — these are 124M
base models):

  * ``mention``  — a passage that NAMES a weekday X, then a probe query whose
    next token should be a weekday. Two query kinds:
      - ``recall``  (kq=0): the answer repeats X ("born on a Friday … born on a").
      - ``derived`` (kq!=0): the answer is X shifted by kq days in CALENDAR order
        ("meeting on Tuesday … the day right after is a" -> Wednesday).
    The injection fires on X's token (the gemma weekday probe), so the
    counterfactual "swap X's pattern to Y" pits the TEXT token (still literally
    "Friday") against the INJECTED day Y. text_answer = day_plus(X, kq);
    the swap's counterfactual answer = day_plus(Y, kq).

  * ``implant`` — a DAY-FREE passage with a neutral noun, then the same style of
    query. No weekday is present, so clean acts are exactly zero; a transform
    injects day Y onto the neutral noun's tokens. Tests whether a spurious
    injection conjures a day belief from nothing. cf_answer = day_plus(Y, kq).

THE STORE-vs-CALENDAR TRAP (map explicitly):
  * STORE order == the r=7 activation-channel order the site/source use, ==
    kaushikreddyxyz/climbmix-scored name-sorted weekday columns 47..53:
        friday, monday, saturday, sunday, thursday, tuesday, wednesday
    -> channel index of a day = STORE_ORDER.index(day).
  * CALENDAR order == the semantic sequence day arithmetic uses:
        monday, tuesday, wednesday, thursday, friday, saturday, sunday
    -> day_plus() and every "day after / before" query live here.
  A onehot injection for day Y sets channel STORE_ORDER.index(Y); the readout for
  a day answer indexes CALENDAR_ORDER. Conflating the two silently mislabels
  every counterfactual — assert both mappings (see test_causal.py).

Pure module: stdlib only, no torch / no harness. ``generate_items()`` is a
deterministic function of (NAMES, contexts, days).
"""
from __future__ import annotations

# r=7 activation-channel order (site/source order). Channel index := index here.
STORE_ORDER = ["friday", "monday", "saturday", "sunday",
               "thursday", "tuesday", "wednesday"]
# Semantic weekday sequence for day arithmetic and answer readout.
CALENDAR_ORDER = ["monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday"]

assert sorted(STORE_ORDER) == sorted(CALENDAR_ORDER), "day set mismatch"


def store_idx(day: str) -> int:
    """Activation-channel index of a weekday (STORE order)."""
    return STORE_ORDER.index(day)


def cal_idx(day: str) -> int:
    """Calendar index (0=monday .. 6=sunday) for day arithmetic + readout."""
    return CALENDAR_ORDER.index(day)


def day_plus(day: str, k: int) -> str:
    """The weekday k days after ``day`` in CALENDAR order (k may be negative)."""
    return CALENDAR_ORDER[(cal_idx(day) + k) % 7]


def display(day: str) -> str:
    """Surface form used in prompts and in the ' <Day>' completion."""
    return day.capitalize()


def completion(day: str) -> str:
    """The completion string scored at the answer position (leading space:
    every prompt ends with the article 'a', so the day is ' Monday' etc.)."""
    return " " + display(day)


# --------------------------------------------------------------------------- #
# Names x contexts == the ~40 mention templates.
# --------------------------------------------------------------------------- #
NAMES = ["Mark", "Sarah", "David", "Elena", "James"]

# Each entry: (template_id, kind, kq, builder). builder(name, Day) -> prompt that
# ENDS right before the answer weekday (last token is the article 'a'). Day is the
# capitalized MENTIONED weekday. kq is the calendar offset from X to the answer.
_MENTION_CONTEXTS = [
    ("R1", "recall", 0,
     lambda n, d: f"{n} was born on a {d}. When asked, {n} said the birth was on a"),
    ("R2", "recall", 0,
     lambda n, d: f"{n} got married on a {d}. According to the records, {n}'s wedding was on a"),
    ("R3", "recall", 0,
     lambda n, d: f"{n} started the new job on a {d}. {n} clearly remembers starting on a"),
    ("R4", "recall", 0,
     lambda n, d: f"{n} arrived in town on a {d}. Everyone agrees {n} arrived on a"),
    ("R5", "recall", 0,
     lambda n, d: f"{n} had the interview on a {d}. The interview with {n} took place on a"),
    ("D1", "derived", 1,
     lambda n, d: f"{n} has a meeting on {d}. The day right after {n}'s meeting is a"),
    ("D2", "derived", -1,
     lambda n, d: f"{n} leaves on {d}. The day just before {n} leaves is a"),
    ("D3", "derived", 2,
     lambda n, d: f"{n} started a fast on {d}. Two days after {n} started is a"),
]


def _implant_specs(name: str):
    """(template_id, kq, prompt, noun, char_start, char_end) for the day-free
    implant contexts. The noun is space-delimited so BPE tokenizes it as its own
    group -> prefix-additive char->token span (see noun_token_span)."""
    out = []
    # I1: neutral noun "town".
    prefix = f"{name} was born in a small"
    noun = " town"
    suffix = ". When asked which day, the answer is a"
    out.append(("I1", 0, prefix + noun + suffix, noun.strip(),
                len(prefix), len(prefix) + len(noun)))
    # I2: neutral noun "car".
    prefix = f"{name} bought a used"
    noun = " car"
    suffix = ". Asked which day it happened, the answer is a"
    out.append(("I2", 0, prefix + noun + suffix, noun.strip(),
                len(prefix), len(prefix) + len(noun)))
    return out


def noun_token_span(encode_fn, prompt: str, char_start: int, char_end: int):
    """Token span [start, end) covering ``prompt[char_start:char_end]``, computed
    by prefix length: start = len(encode(prompt[:char_start])), end =
    len(encode(prompt[:char_end])). Exact when the substring sits on tokenizer
    split boundaries (the implant nouns do: space-delimited). ``encode_fn`` is the
    nanochat tokenizer's encode (BOS-free)."""
    start = len(encode_fn(prompt[:char_start]))
    end = len(encode_fn(prompt[:char_end]))
    if end <= start:  # degenerate boundary (shouldn't happen for the space-delimited nouns)
        end = start + 1
    return start, end


def generate_items():
    """Deterministic list of item dicts. Fields:

      mention: id, family='mention', kind, template, name, day (=X mentioned),
               kq, prompt, text_answer (=day_plus(X, kq)).
      implant: id, family='implant', kind='recall', template, name,
               inject_day (=Y), kq, prompt, noun, noun_char (start,end),
               text_answer=None.
    """
    items = []
    for name in NAMES:
        for tid, kind, kq, build in _MENTION_CONTEXTS:
            for x in CALENDAR_ORDER:  # mentioned day
                items.append({
                    "id": f"mention/{kind}/{tid}/{name}/{x}",
                    "family": "mention", "kind": kind, "template": tid,
                    "name": name, "day": x, "kq": kq,
                    "prompt": build(name, display(x)),
                    "text_answer": day_plus(x, kq),
                })
    for name in NAMES:
        for tid, kq, prompt, noun, cs, ce in _implant_specs(name):
            for y in CALENDAR_ORDER:  # injected day
                items.append({
                    "id": f"implant/recall/{tid}/{name}/{y}",
                    "family": "implant", "kind": "recall", "template": tid,
                    "name": name, "inject_day": y, "kq": kq,
                    "prompt": prompt, "noun": noun, "noun_char": (cs, ce),
                    "text_answer": None,
                })
    return items


if __name__ == "__main__":
    its = generate_items()
    n_m = sum(1 for i in its if i["family"] == "mention")
    n_i = sum(1 for i in its if i["family"] == "implant")
    print(f"{len(its)} items: {n_m} mention ({len(NAMES)} names x "
          f"{len(_MENTION_CONTEXTS)} contexts x 7 days), {n_i} implant")
    for i in its[:3] + its[n_m:n_m + 2]:
        print(" ", i["id"], "->", repr(i["prompt"]))
