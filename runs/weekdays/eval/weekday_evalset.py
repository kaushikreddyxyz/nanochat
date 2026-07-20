"""Deterministic template generator for weekday_v1 (422 completion-style items:
cyclic order next/prev/k-step, position-in-week, weekend/usage, sanity), scored by
length-normalized CE per option in run_evals.py. Same bytes every run; meta.fill_in
marks items whose answer must NOT appear in the prompt (leak-tested).
Regenerate: python runs/weekdays/eval/weekday_evalset.py
"""
import argparse
import json
import os

# Calendar order (Monday-first). This is the human-facing option order; it is
# unrelated to the store's name-sorted channel order (friday,monday,...), which
# only matters inside the injection source.
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKEND = ["Saturday", "Sunday"]
WORKWEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
KWORDS = {2: "Two", 3: "Three", 4: "Four"}
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth",
            5: "fifth", 6: "sixth", 7: "seventh"}

CATEGORIES = ["order_next", "order_prev", "order_k", "fact_position",
              "usage_weekend", "usage_context", "sanity"]

# Categories whose answers are one of the 7 weekdays AND are perfectly balanced
# across the week by construction (each day is the answer the same number of
# times). test_evalset checks per-day balance over exactly these.
BALANCED_DAY_CATEGORIES = ["order_next", "order_prev", "order_k", "usage_context"]


def _next(d, n=1):
    return WEEKDAYS[(WEEKDAYS.index(d) + n) % 7]


def _prev(d, n=1):
    return WEEKDAYS[(WEEKDAYS.index(d) - n) % 7]


# --------------------------------------------------------------------------- #
# Surface templates. Each entry: (template_id, prompt_format). Multiple per
# category so the set is not a single memorizable pattern.
# --------------------------------------------------------------------------- #
NEXT_TEMPLATES = [
    ("next_a", "The day after {D} is"),
    ("next_b", "The day that comes after {D} is"),
    ("next_c", "If today is {D}, then tomorrow is"),
    ("next_d", "One day after {D} comes"),
    ("next_e", "{D} is immediately followed by"),
    ("next_f", "Moving forward one day from {D}, you reach"),
    ("next_g", "The weekday right after {D} is"),
    ("next_h", "After {D} comes"),
    ("next_i", "The next day after {D} will be"),
    ("next_j", "Once {D} is over, the following day is"),
]
PREV_TEMPLATES = [
    ("prev_a", "The day before {D} is"),
    ("prev_b", "The day that comes before {D} is"),
    ("prev_c", "If today is {D}, then yesterday was"),
    ("prev_d", "One day before {D} comes"),
    ("prev_e", "{D} is immediately preceded by"),
    ("prev_f", "Moving back one day from {D}, you reach"),
    ("prev_g", "The weekday right before {D} is"),
    ("prev_h", "Before {D} comes"),
    ("prev_i", "The previous day before {D} was"),
    ("prev_j", "Just before {D} comes"),
]
# order_k phrasings, split by direction so the answer computation is unambiguous.
KAFTER_TEMPLATES = [
    ("kaft_a", "{Kw} days after {D} is"),
    ("kaft_b", "The day {Kw} days after {D} is"),
    ("kaft_c", "Counting {Kw} days forward from {D}, you reach"),
    ("kaft_d", "Going {Kw} days ahead of {D}, the day is"),
]
KBEFORE_TEMPLATES = [
    ("kbef_a", "{Kw} days before {D} is"),
    ("kbef_b", "The day {Kw} days before {D} is"),
    ("kbef_c", "Counting {Kw} days back from {D}, you reach"),
    ("kbef_d", "Going {Kw} days behind {D}, the day is"),
]
# fact_position: the convention is STATED in-prompt to remove ambiguity.
# {start} names the first day; {phrase} asks for the day at position n (2..7 —
# position 1 is excluded because the answer would equal the named start day,
# leaking it into the prompt).
FACT_PHRASES = [
    ("fact_ord", "the {ORD} day of the week is"),
    ("fact_num", "day number {N} of the week is"),
    ("fact_pos", "the day in position {N}, counting from the start, is"),
]
FACT_CONVENTIONS = [("mon", "Monday"), ("sun", "Sunday")]

# usage_context: self-contained, in-prompt day-fact. The first four are
# association/copy (answer == the stated day, fill_in False by design — this is
# the task's canonical "market held every Wednesday -> market day is Wednesday").
# The last is an INFERENCE variant (answer = the day before; leak-free, fill_in
# True) so the category also exercises reasoning, not only copying.
CONTEXT_TEMPLATES = [
    ("ctx_market", "In a town where the market is held every {D}, the market day is", "same"),
    ("ctx_shop", "A shop that closes every {D} has its weekly closing day on", "same"),
    ("ctx_club", "The book club that meets each {D} holds its meeting on", "same"),
    ("ctx_water", "She waters the plants every {D}, so her watering day is", "same"),
    ("ctx_bake", "The bakery bakes fresh bread every {D}; its baking day is", "same"),
    ("ctx_setup", "The market is held every {D}, and traders set up the evening before, which is", "prev"),
]

# sanity: trivial non-weekday completions to calibrate on-distribution behavior.
# (answer, options) — answer must not appear in the prompt.
SANITY_ITEMS = [
    ("On a clear day the sky is", "blue", ["blue", "green", "purple", "orange"]),
    ("Fresh grass is usually colored", "green", ["green", "blue", "red", "yellow"]),
    ("Freshly fallen snow is", "white", ["white", "black", "green", "red"]),
    ("The opposite of hot is", "cold", ["cold", "loud", "tall", "fast"]),
    ("The opposite of up is", "down", ["down", "left", "warm", "soft"]),
    ("The opposite of big is", "small", ["small", "loud", "bright", "heavy"]),
    ("Two plus two equals", "four", ["four", "three", "five", "seven"]),
    ("Three plus one equals", "four", ["four", "two", "six", "nine"]),
    ("The first letter of the alphabet is", "A", ["A", "B", "M", "Z"]),
    ("The color of a ripe banana is", "yellow", ["yellow", "blue", "purple", "gray"]),
    ("A cat says", "meow", ["meow", "woof", "moo", "oink"]),
    ("A dog says", "woof", ["woof", "meow", "quack", "neigh"]),
    ("Water freezes into", "ice", ["ice", "steam", "sand", "smoke"]),
    ("The sun rises in the", "east", ["east", "west", "north", "middle"]),
    ("A week has this many days:", "seven", ["seven", "five", "ten", "three"]),
    ("The opposite of day is", "night", ["night", "noon", "week", "hour"]),
    ("Ice is very", "cold", ["cold", "hot", "loud", "sweet"]),
    ("Fire is very", "hot", ["hot", "cold", "quiet", "blue"]),
    ("The number after nine is", "ten", ["ten", "eight", "twelve", "one"]),
    ("A triangle has this many sides:", "three", ["three", "four", "five", "two"]),
    ("The opposite of open is", "closed", ["closed", "green", "fast", "round"]),
    ("Lemons taste", "sour", ["sour", "sweet", "salty", "spicy"]),
    ("The opposite of empty is", "full", ["full", "cold", "near", "loud"]),
    ("A baby dog is called a", "puppy", ["puppy", "kitten", "foal", "calf"]),
]


def _item(items, category, template, prompt, options, answer, meta):
    """Append a fully-formed item with a stable per-category id."""
    n = sum(1 for it in items if it["category"] == category)
    it = {
        "id": f"{category}_{n:03d}",
        "category": category,
        "prompt": prompt,
        "options": list(options),
        "answer": answer,
        "meta": {"template": template, **meta},
    }
    assert answer in it["options"], (it["id"], answer, it["options"])
    items.append(it)
    return it


def generate_items():
    """Deterministic list[dict] of all eval items (fixed order)."""
    items = []

    # order_next / order_prev — each template covers all 7 days => balanced.
    for tid, tmpl in NEXT_TEMPLATES:
        for d in WEEKDAYS:
            _item(items, "order_next", tid, tmpl.format(D=d), WEEKDAYS, _next(d),
                  {"day": d, "direction": "next", "k": 1, "fill_in": True})
    for tid, tmpl in PREV_TEMPLATES:
        for d in WEEKDAYS:
            _item(items, "order_prev", tid, tmpl.format(D=d), WEEKDAYS, _prev(d),
                  {"day": d, "direction": "prev", "k": 1, "fill_in": True})

    # order_k — k in {2,3,4}, both directions, wraps (tests cyclic structure).
    # +k / -k mod 7 is a bijection over the week => each (k,dir,template) is balanced.
    for k in (2, 3, 4):
        for tid, tmpl in KAFTER_TEMPLATES:
            for d in WEEKDAYS:
                _item(items, "order_k", f"{tid}_k{k}", tmpl.format(Kw=KWORDS[k], D=d),
                      WEEKDAYS, _next(d, k),
                      {"day": d, "direction": "after", "k": k, "fill_in": True})
        for tid, tmpl in KBEFORE_TEMPLATES:
            for d in WEEKDAYS:
                _item(items, "order_k", f"{tid}_k{k}", tmpl.format(Kw=KWORDS[k], D=d),
                      WEEKDAYS, _prev(d, k),
                      {"day": d, "direction": "before", "k": k, "fill_in": True})

    # fact_position — in-prompt convention, positions 2..7 (pos 1 == start day,
    # excluded to stay leak-free). Answer is Monday-indexed off the stated start.
    for cid, start in FACT_CONVENTIONS:
        # order[pos-1] is the day at 1-based position `pos` given this start day.
        order = [WEEKDAYS[(WEEKDAYS.index(start) + i) % 7] for i in range(7)]
        for pid, phrase in FACT_PHRASES:
            for pos in range(2, 8):
                ph = phrase.format(ORD=ORDINALS[pos], N=pos)
                prompt = f"If the week starts on {start}, {ph}"
                ans = order[pos - 1]
                _item(items, "fact_position", f"{cid}_{pid}_p{pos}", prompt,
                      WEEKDAYS, ans,
                      {"convention": start, "position": pos, "day": ans, "fill_in": True})

    # usage_weekend — pair completion (fill_in, leak-free) + MC classification
    # (options listed in-prompt, fill_in False).
    _item(items, "usage_weekend", "pair_sat",
          "The two days of the weekend are Saturday and",
          WEEKDAYS, "Sunday", {"kind": "pair", "fill_in": True})
    _item(items, "usage_weekend", "pair_sun",
          "The two days of the weekend are Sunday and",
          WEEKDAYS, "Saturday", {"kind": "pair", "fill_in": True})
    # "which of these is NOT part of the weekend" — answer is each weekday.
    for w in WORKWEEK:
        opts = [w, "Saturday", "Sunday"]
        _item(items, "usage_weekend", f"mc_notweekend_{w.lower()}",
              f"Among {w}, Saturday, and Sunday, the day that is not part of the weekend is",
              opts, w, {"kind": "mc_not_weekend", "fill_in": False})
    # "which of these IS the weekend day" — two distinct weekday distractors plus
    # the weekend answer (alternating Saturday/Sunday); options unique by construction.
    for i, w in enumerate(WORKWEEK):
        wknd = "Saturday" if i % 2 == 0 else "Sunday"
        distractor2 = WORKWEEK[(i + 1) % 5]
        opts = [w, distractor2, wknd]
        _item(items, "usage_weekend", f"mc_isweekend_{w.lower()}",
              f"Among {opts[0]}, {opts[1]}, and {opts[2]}, the weekend day is",
              opts, wknd, {"kind": "mc_is_weekend", "fill_in": False})

    # usage_context — self-contained in-prompt fact; association (answer==day) or
    # inference (answer==prev day). Each template covers all 7 days => balanced.
    for tid, tmpl, mode in CONTEXT_TEMPLATES:
        for d in WEEKDAYS:
            ans = d if mode == "same" else _prev(d)
            _item(items, "usage_context", tid, tmpl.format(D=d), WEEKDAYS, ans,
                  {"day": d, "mode": mode, "fill_in": mode != "same"})

    # sanity — trivial non-weekday calibration.
    for i, (prompt, ans, opts) in enumerate(SANITY_ITEMS):
        _item(items, "sanity", f"s{i:02d}", prompt, opts, ans,
              {"fill_in": True, "weekday": False})

    return items


def leaks_answer(item):
    """True iff the answer word appears in the prompt (case-insensitive whole-word
    match). Only meaningful for fill_in items; association/MC items legitimately
    name a day in the prompt."""
    import re
    return re.search(rf"\b{re.escape(item['answer'])}\b", item["prompt"], re.IGNORECASE) is not None


def to_jsonl_bytes(items):
    """Deterministic JSONL encoding (stable key order, no trailing newline drift)."""
    lines = [json.dumps(it, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
             for it in items]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_evalset(path):
    items = generate_items()
    data = to_jsonl_bytes(items)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return items


def category_counts(items):
    out = {}
    for it in items:
        out[it["category"]] = out.get(it["category"], 0) + 1
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "evalsets", "weekday_v1.jsonl")
    ap = argparse.ArgumentParser(description="Generate the weekday_v1 eval set.")
    ap.add_argument("--out", default=default_out, help="output JSONL path")
    args = ap.parse_args()
    items = write_evalset(args.out)
    counts = category_counts(items)
    print(f"Wrote {len(items)} items -> {args.out}")
    for c in CATEGORIES:
        print(f"  {c:16s} {counts.get(c, 0)}")


if __name__ == "__main__":
    main()
