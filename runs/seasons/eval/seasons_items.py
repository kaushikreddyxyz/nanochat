"""Deterministic causal/counterfactual item set: ``mention`` items name season X in text
(recall kq=0 / derived kq!=0 queries; swap-to-Y pits text against injection), ``implant``
items are season-free (clean acts exactly zero; a transform injects Y on a neutral noun).
Channel index = STORE order; season arithmetic/readout = CYCLE order.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seasons_concepts import CYCLE_ORDER, FAMILY, STORE_ORDER, season_plus  # noqa: E402

# Interface the shared causal driver (runs/lib/eval/causal.py, via --items-module)
# reads to stay family-agnostic: the item key holding the mentioned / injected label.
LABEL_FIELD = "season"
INJECT_FIELD = "inject_season"

NAMES = ["Mark", "Sarah", "David", "Elena", "James"]


def store_idx(season):
    """Activation-channel index of a season (STORE order)."""
    return FAMILY.store_row(season)


def cycle_idx(season):
    """Cycle index (0=spring .. 3=winter) for season arithmetic and answer readout."""
    return FAMILY.cycle_position(season)


def completion(season):
    """The string scored at the answer position. Every prompt ends with a preposition
    or copula, so the completion is ' spring' etc. — seasons are not proper nouns, so
    the surface form stays lowercase everywhere."""
    return " " + season


# Each entry: (template_id, kind, kq, builder). builder(name, season) -> prompt that
# ENDS right before the answer season. kq is the cycle offset from the mentioned
# season X to the answer; on a 4-cycle kq=2 is the antipode.
_MENTION_CONTEXTS = [
    ("R1", "recall", 0,
     lambda n, s: f"{n} was born in {s}. When asked, {n} said the birth was in"),
    ("R2", "recall", 0,
     lambda n, s: f"{n} got married in {s}. According to the records, the wedding was in"),
    ("R3", "recall", 0,
     lambda n, s: f"{n} started the new job in {s}. {n} clearly remembers starting in"),
    ("R4", "recall", 0,
     lambda n, s: f"{n} arrived in town in {s}. Everyone agrees {n} arrived in"),
    ("R5", "recall", 0,
     lambda n, s: f"{n} had the interview in {s}. The interview with {n} took place in"),
    ("D1", "derived", 1,
     lambda n, s: f"{n} has a wedding in {s}. The season right after {n}'s wedding is"),
    ("D2", "derived", -1,
     lambda n, s: f"{n} leaves in {s}. The season just before {n} leaves is"),
    ("D3", "derived", 2,
     lambda n, s: f"{n} began the project in {s}. Two seasons after {n} began is"),
]


def _implant_specs(name):
    """(template_id, kq, prompt, noun, char_start, char_end) for the season-free
    implant contexts. The noun is space-delimited so BPE tokenizes it as its own group
    -> prefix-additive char->token span (see noun_token_span)."""
    out = []
    prefix = f"{name} was born in a small"
    noun = " town"
    suffix = ". Asked which season it happened in, the answer is"
    out.append(("I1", 0, prefix + noun + suffix, noun.strip(),
                len(prefix), len(prefix) + len(noun)))
    prefix = f"{name} bought a used"
    noun = " car"
    suffix = ". Asked which season the purchase happened in, the answer is"
    out.append(("I2", 0, prefix + noun + suffix, noun.strip(),
                len(prefix), len(prefix) + len(noun)))
    return out


def noun_token_span(encode_fn, prompt, char_start, char_end):
    """Token span [start, end) covering ``prompt[char_start:char_end]``, computed by
    prefix length. Exact when the substring sits on tokenizer split boundaries (the
    implant nouns do: space-delimited). ``encode_fn`` is the nano tokenizer's
    BOS-free encode."""
    start = len(encode_fn(prompt[:char_start]))
    end = len(encode_fn(prompt[:char_end]))
    if end <= start:
        end = start + 1
    return start, end


def generate_items():
    """Deterministic list of item dicts. Fields:

      mention: id, family='mention', kind, template, name, season (=X mentioned), kq,
               prompt, text_answer (=season_plus(X, kq)).
      implant: id, family='implant', kind='recall', template, name, inject_season (=Y),
               kq, prompt, noun, noun_char (start,end), text_answer=None.
    """
    items = []
    for name in NAMES:
        for tid, kind, kq, build in _MENTION_CONTEXTS:
            for x in CYCLE_ORDER:
                items.append({
                    "id": f"mention/{kind}/{tid}/{name}/{x}",
                    "family": "mention", "kind": kind, "template": tid,
                    "name": name, "season": x, "kq": kq,
                    "prompt": build(name, x),
                    "text_answer": season_plus(x, kq),
                })
    for name in NAMES:
        for tid, kq, prompt, noun, cs, ce in _implant_specs(name):
            for y in CYCLE_ORDER:
                items.append({
                    "id": f"implant/recall/{tid}/{name}/{y}",
                    "family": "implant", "kind": "recall", "template": tid,
                    "name": name, "inject_season": y, "kq": kq,
                    "prompt": prompt, "noun": noun, "noun_char": (cs, ce),
                    "text_answer": None,
                })
    return items


if __name__ == "__main__":
    its = generate_items()
    n_m = sum(1 for i in its if i["family"] == "mention")
    n_i = sum(1 for i in its if i["family"] == "implant")
    print(f"{len(its)} items: {n_m} mention ({len(NAMES)} names x "
          f"{len(_MENTION_CONTEXTS)} contexts x {len(CYCLE_ORDER)} seasons), {n_i} implant")
    print(f"  store order {STORE_ORDER} != cycle order {CYCLE_ORDER}")
    for i in its[:3] + its[n_m:n_m + 2]:
        print(" ", i["id"], "->", repr(i["prompt"]))
