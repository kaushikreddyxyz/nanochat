"""Deterministic template generator for seasons_v1 (completion-style items: cyclic
order, month<->season by hemisphere, characteristics, solstice/equinox, the 'fall'
surface form, and a polysemy control), scored by length-normalized CE per option in
run_evals.py. Same bytes every run; meta.fill_in marks answers that must NOT appear
in the prompt; meta.hemisphere is non-null only where the answer depends on it.
Regenerate: python runs/seasons/eval/seasons_evalset.py
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seasons_concepts import CYCLE_ORDER, season_plus  # noqa: E402

SEASONS = list(CYCLE_ORDER)  # human-facing option order == the seasonal cycle
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
KWORDS = {2: "Two", 3: "Three"}
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}

CATEGORIES = ["order_next", "order_prev", "order_k", "fact_position",
              "month_to_season", "season_to_month", "characteristic",
              "usage_context", "solstice_equinox", "synonym_fall",
              "polysemy", "sanity"]

# Categories whose answers are seasons AND are perfectly balanced across the cycle by
# construction (each season answers the same number of times). fact_position is NOT
# balanced (position 1 is dropped as an answer leak, which unbalances the rest).
BALANCED_SEASON_CATEGORIES = ["order_next", "order_prev", "order_k",
                              "month_to_season", "characteristic", "usage_context"]

# Meteorological three-month seasons, northern hemisphere. The southern table is the
# antipode (cycle +2), so every month maps to opposite seasons in the two hemispheres —
# this is the whole reason meta.hemisphere exists.
NORTHERN_SEASON_MONTHS = {
    "spring": ["March", "April", "May"],
    "summer": ["June", "July", "August"],
    "autumn": ["September", "October", "November"],
    "winter": ["December", "January", "February"],
}
SEASON_MONTHS = {
    "northern": NORTHERN_SEASON_MONTHS,
    "southern": {s: NORTHERN_SEASON_MONTHS[season_plus(s, 2)] for s in SEASONS},
}
MONTH_SEASON = {h: {m: s for s, ms in SEASON_MONTHS[h].items() for m in ms}
                for h in ("northern", "southern")}


# --------------------------------------------------------------------------- #
# Surface templates. Season words are never sentence-initial, so every season
# mention (prompt and option alike) stays lowercase.
# --------------------------------------------------------------------------- #
NEXT_TEMPLATES = [
    ("next_a", "The season after {S} is"),
    ("next_b", "The season that comes after {S} is"),
    ("next_c", "If it is {S} now, the next season will be"),
    ("next_d", "One season after {S} comes"),
    ("next_e", "The season immediately following {S} is"),
    ("next_f", "Moving forward one season from {S}, you reach"),
    ("next_g", "The season right after {S} is"),
    ("next_h", "Coming after {S} is"),
    ("next_i", "The next season after {S} will be"),
    ("next_j", "Once {S} is over, the following season is"),
    ("next_k", "Following {S} on the calendar comes"),
    ("next_l", "The season that begins when {S} ends is"),
]
PREV_TEMPLATES = [
    ("prev_a", "The season before {S} is"),
    ("prev_b", "The season that comes before {S} is"),
    ("prev_c", "If it is {S} now, the previous season was"),
    ("prev_d", "One season before {S} comes"),
    ("prev_e", "The season immediately preceding {S} is"),
    ("prev_f", "Moving back one season from {S}, you reach"),
    ("prev_g", "The season right before {S} is"),
    ("prev_h", "Coming before {S} is"),
    ("prev_i", "The previous season before {S} was"),
    ("prev_j", "Just before {S} comes"),
    ("prev_k", "Preceding {S} on the calendar is"),
    ("prev_l", "The season that ends when {S} begins is"),
]
# order_k phrasings, split by direction. On a 4-cycle k=2 is the antipode (the two
# directions share an answer) and k=3 mirrors k=1 — meta.k/meta.direction keep the
# two arithmetic paths separable in analysis.
KAFTER_TEMPLATES = [
    ("kaft_a", "{Kw} seasons after {S} is"),
    ("kaft_b", "The season {Kw} seasons after {S} is"),
    ("kaft_c", "Counting {Kw} seasons forward from {S}, you reach"),
    ("kaft_d", "Going {Kw} seasons ahead of {S}, the season is"),
]
KBEFORE_TEMPLATES = [
    ("kbef_a", "{Kw} seasons before {S} is"),
    ("kbef_b", "The season {Kw} seasons before {S} is"),
    ("kbef_c", "Counting {Kw} seasons back from {S}, you reach"),
    ("kbef_d", "Going {Kw} seasons behind {S}, the season is"),
]

# fact_position: the starting season is STATED in-prompt. Positions 2..4 only —
# position 1 would answer with the named start season, leaking it into the prompt.
FACT_PHRASES = [
    ("fact_ord", "the {ORD} season of the year is"),
    ("fact_num", "season number {N} of the year is"),
    ("fact_pos", "the season in position {N}, counting from the start, is"),
]
FACT_CONVENTIONS = ["spring", "winter"]

# month_to_season: (template_id, hemisphere, hemisphere_stated, format). A southern
# item MUST name its hemisphere or it is unanswerable; northern items include one
# unstated variant to measure the model's default.
MONTH_SEASON_TEMPLATES = [
    ("m2s_north_a", "northern", True, "In the northern hemisphere, {M} belongs to the season of"),
    ("m2s_north_b", "northern", True,
     "Grouping the year into three-month seasons in the northern hemisphere, {M} is counted as part of"),
    ("m2s_north_c", "northern", False, "The month of {M} is part of the season called"),
    ("m2s_south_a", "southern", True, "In the southern hemisphere, {M} belongs to the season of"),
    ("m2s_south_b", "southern", True, "In Australia, the month of {M} is part of the season called"),
]

# season_to_month: (template_id, hemisphere, part, format). part selects which of the
# season's three months is the answer; "last" lists the other two in the prompt.
SEASON_MONTH_TEMPLATES = [
    ("s2m_north_last", "northern", "last",
     "In the northern hemisphere, the three months of {S} are {M1}, {M2}, and"),
    ("s2m_north_first", "northern", "first",
     "Grouping the year into three-month seasons in the northern hemisphere, {S} begins in the month of"),
    ("s2m_north_middle", "northern", "middle",
     "In the northern hemisphere, the middle month of {S} is"),
    ("s2m_south_middle", "southern", "middle",
     "In the southern hemisphere, the middle month of {S} is"),
]

# characteristic: weather/activity descriptions that pick out a season in EITHER
# hemisphere (the description travels with the season, not the calendar) — so these
# carry no hemisphere. Exactly 5 per season keeps the category balanced.
CHARACTERISTIC_ITEMS = [
    ("char_bud", "spring", "The season when flower buds open and baby animals are born is"),
    ("char_leaf", "spring", "The season when bare trees grow fresh green leaves again is"),
    ("char_sow", "spring", "The season when farmers sow their seeds once the frosts have ended is"),
    ("char_thaw", "spring", "The season when the thaw swells the rivers with meltwater is"),
    ("char_nest", "spring", "The season of nesting birds and early blossom is"),
    ("char_long", "summer", "The season with the longest days and the hottest weather is"),
    ("char_school", "summer", "The season when schools close for their long holiday is"),
    ("char_beach", "summer", "The season when crowds swim at the beach to escape the heat is"),
    ("char_cream", "summer", "The season when ice cream sells fastest and fans run all day is"),
    ("char_evening", "summer", "The season of sunburn, lemonade, and long light evenings is"),
    ("char_red", "autumn", "The season when leaves turn red and drop from the trees is"),
    ("char_harvest", "autumn", "The season when farmers gather the ripe harvest from the fields is"),
    ("char_rake", "autumn", "The season when the days shorten and people rake dead leaves is"),
    ("char_squirrel", "autumn", "The season when squirrels store nuts before the cold arrives is"),
    ("char_pumpkin", "autumn", "The season of pumpkins, apple picking, and misty mornings is"),
    ("char_freeze", "winter", "The season when lakes freeze over and snow covers the ground is"),
    ("char_short", "winter", "The season with the shortest days and the coldest nights is"),
    ("char_bear", "winter", "The season when bears hibernate and the trees stand bare is"),
    ("char_coat", "winter", "The season when people wear heavy coats, scarves, and gloves is"),
    ("char_frost", "winter", "The season of frost, icicles, and hot drinks by the fire is"),
]

# usage_context: self-contained in-prompt fact. The first five are association/copy
# (answer == the stated season, fill_in False by design); the last is an INFERENCE
# variant (answer = the previous season; leak-free) so the category also tests
# reasoning, not only copying.
CONTEXT_TEMPLATES = [
    ("ctx_festival", "In a village where the harvest festival is held every {S}, the festival season is", "same"),
    ("ctx_shop", "A beach shop that opens only in {S} does all of its business in", "same"),
    ("ctx_league", "The local league that plays its matches through {S} holds its games in", "same"),
    ("ctx_visit", "She visits her grandmother every {S}, so her visiting season is", "same"),
    ("ctx_road", "The mountain road is closed every {S}, so its closure season is", "same"),
    ("ctx_plan", "The fair is held every {S}, and the organizers start planning one season earlier, in", "prev"),
]

# solstice_equinox: the astronomical convention (season starts at the solstice or
# equinox) is stated in-prompt, and each event is asked in both hemispheres.
SOLSTICE_EVENTS = [("june_solstice", "the June solstice", "summer"),
                   ("december_solstice", "the December solstice", "winter"),
                   ("march_equinox", "the March equinox", "spring"),
                   ("september_equinox", "the September equinox", "autumn")]

# synonym_fall: the American surface form of autumn. Season-cycle answers use the
# season options; month answers use the month options.
FALL_ITEMS = [
    ("fall_next", "The season right after fall is", "winter", "season", "northern_none"),
    ("fall_prev", "The season right before fall is", "summer", "season", "northern_none"),
    ("fall_k2", "Two seasons after fall comes", "spring", "season", "northern_none"),
    ("fall_british", "The season that Americans call fall is known in British English as",
     "autumn", "season", "northern_none"),
    ("fall_months_north", "In the northern hemisphere, the three months of fall are "
     "September, October, and", "November", "month", "northern"),
    ("fall_months_south", "In the southern hemisphere, fall covers March, April, and",
     "May", "month", "southern"),
]

# polysemy control — the seasons analogue of a specificity check, and the reason this
# suite is not a copy of the weekday one. Every item's correct answer denotes something
# OTHER than a season (meta.expect_season_concept False), while a season word is
# present in the prompt or is itself the answer, so the lexically-firing probes inject
# on a token where the season reading is wrong.
#   kind "non_season_sense"  — the season word carries a non-season meaning.
#   kind "season_word_answer" — the season reading is correct but the answer is not a
#                               season (a season word in the prompt must not push a
#                               season name out of the model).
# (template, kind, season_word, sense, prompt, answer, options)
POLYSEMY_ITEMS = [
    ("poly_water", "non_season_sense", "spring", "water_source",
     "Water that flows naturally out of the ground is called a",
     "spring", ["spring", "cloud", "desert", "mountain"]),
    ("poly_coil", "non_season_sense", "spring", "coil",
     "A coil of metal that pushes back when you squeeze it is a",
     "spring", ["spring", "magnet", "brick", "rope"]),
    ("poly_sofa", "non_season_sense", "spring", "coil",
     "The old sofa sagged because a broken metal spring poked through the",
     "cushion", ["cushion", "harvest", "glacier", "festival"]),
    ("poly_leap", "non_season_sense", "spring", "jump",
     "The cat crouched low and then made a sudden spring onto the",
     "table", ["table", "month", "holiday", "calendar"]),
    ("poly_mountain", "non_season_sense", "spring", "water_source",
     "The hikers refilled their bottles at a cold mountain spring of fresh",
     "water", ["water", "snowfall", "harvest", "sunlight"]),
    ("poly_step", "non_season_sense", "spring", "idiom",
     "She keeps a spring in her step, which means she walks with plenty of",
     "energy", ["energy", "snow", "paperwork", "silence"]),
    ("poly_jail", "non_season_sense", "spring", "release",
     "His friends planned to spring him from jail, meaning they wanted to help him",
     "escape", ["escape", "study", "apologize", "celebrate"]),
    ("poly_trip", "non_season_sense", "fall", "drop",
     "If you trip near the top of the stairs, you might",
     "fall", ["fall", "fly", "swim", "sing"]),
    ("poly_climber", "non_season_sense", "fall", "drop",
     "The climber lost his grip and took a bad fall onto the rocks, breaking his",
     "leg", ["leg", "harvest", "calendar", "festival"]),
    ("poly_rain", "non_season_sense", "fall", "drop",
     "Rain begins to fall once the clouds grow heavy with",
     "water", ["water", "leaves", "holidays", "months"]),
    ("poly_prices", "non_season_sense", "fall", "decrease",
     "Prices continued to fall until the shop was almost giving its stock",
     "away", ["away", "upward", "colder", "sunnier"]),
    ("poly_coast", "non_season_sense", "summer", "verb_reside",
     "Wealthy families used to summer on the coast, spending the warm months beside the",
     "sea", ["sea", "factory", "library", "desk"]),
    ("poly_cattle", "non_season_sense", "winter", "verb_shelter",
     "Farmers winter their cattle indoors, keeping the animals out of the wind and the",
     "rain", ["rain", "sunshine", "music", "paperwork"]),
    ("poly_coat", "season_word_answer", "winter", "season",
     "A winter coat is made thick so that it keeps you",
     "warm", ["warm", "cold", "wet", "loud"]),
    ("poly_sunset", "season_word_answer", "summer", "season",
     "In summer the days are long, so the sun sets very",
     "late", ["late", "early", "never", "silently"]),
    ("poly_firewood", "season_word_answer", "winter", "season",
     "Before winter arrives, many people store extra firewood so they can stay",
     "warm", ["warm", "cool", "damp", "hungry"]),
    ("poly_shower", "season_word_answer", "spring", "season",
     "After a spring shower the bare ground becomes",
     "wet", ["wet", "frozen", "dusty", "golden"]),
    ("poly_evening", "season_word_answer", "autumn", "season",
     "In autumn the days get shorter, so the evenings arrive",
     "earlier", ["earlier", "later", "never", "louder"]),
    ("poly_drought", "season_word_answer", "summer", "season",
     "A long summer drought leaves the riverbed completely",
     "dry", ["dry", "flooded", "frozen", "crowded"]),
    ("poly_roads", "season_word_answer", "winter", "season",
     "Winter storms make the mountain roads dangerously",
     "icy", ["icy", "sunny", "dusty", "warm"]),
    ("poly_seeds", "season_word_answer", "spring", "season",
     "Spring rain helps the newly planted seeds to",
     "grow", ["grow", "freeze", "shatter", "melt"]),
    ("poly_barn", "season_word_answer", "autumn", "season",
     "In autumn the farmer's barn slowly fills with the gathered",
     "grain", ["grain", "snow", "icicles", "seedlings"]),
]

# sanity: trivial non-seasonal completions to calibrate on-distribution behavior.
# Deliberately free of season words, month names and weekday names.
SANITY_ITEMS = [
    ("On a clear day the sky is", "blue", ["blue", "green", "purple", "orange"]),
    ("Fresh grass is usually colored", "green", ["green", "blue", "red", "yellow"]),
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
    ("The opposite of day is", "night", ["night", "noon", "hour", "minute"]),
    ("Fire is very", "hot", ["hot", "cold", "quiet", "blue"]),
    ("The number after nine is", "ten", ["ten", "eight", "twelve", "one"]),
    ("A triangle has this many sides:", "three", ["three", "four", "five", "two"]),
    ("The opposite of open is", "closed", ["closed", "green", "fast", "round"]),
    ("Lemons taste", "sour", ["sour", "sweet", "salty", "spicy"]),
    ("A baby dog is called a", "puppy", ["puppy", "kitten", "foal", "calf"]),
]


def answer_kind(answer):
    """Lexical class of an answer string: 'season', 'month' or 'other'. Purely about
    the WORD — 'spring' as a water source still classifies as 'season' — so it is kept
    separate from meta.expect_season_concept, which is about the meaning."""
    low = answer.lower()
    if low in SEASONS:
        return "season"
    if answer in MONTHS:
        return "month"
    return "other"


def _item(items, category, template, prompt, options, answer, meta):
    """Append a fully-formed item with a stable per-category id. meta must carry
    fill_in, hemisphere, hemisphere_stated and expect_season_concept."""
    n = sum(1 for it in items if it["category"] == category)
    it = {
        "id": f"{category}_{n:03d}",
        "category": category,
        "prompt": prompt,
        "options": list(options),
        "answer": answer,
        "meta": {"template": template, "answer_kind": answer_kind(answer), **meta},
    }
    assert answer in it["options"], (it["id"], answer, it["options"])
    for key in ("fill_in", "hemisphere", "hemisphere_stated", "expect_season_concept"):
        assert key in it["meta"], (it["id"], f"missing meta.{key}")
    items.append(it)
    return it


def _meta(fill_in, expect_season_concept=True, hemisphere=None, hemisphere_stated=False, **rest):
    """meta defaults. hemisphere is None wherever the answer does not depend on it
    (season arithmetic is hemisphere-invariant; only the month tables flip)."""
    return {"fill_in": fill_in, "expect_season_concept": expect_season_concept,
            "hemisphere": hemisphere, "hemisphere_stated": hemisphere_stated, **rest}


def generate_items():
    """Deterministic list[dict] of all eval items (fixed order)."""
    items = []

    for tid, tmpl in NEXT_TEMPLATES:
        for s in SEASONS:
            _item(items, "order_next", tid, tmpl.format(S=s), SEASONS, season_plus(s, 1),
                  _meta(True, season=s, direction="next", k=1))
    for tid, tmpl in PREV_TEMPLATES:
        for s in SEASONS:
            _item(items, "order_prev", tid, tmpl.format(S=s), SEASONS, season_plus(s, -1),
                  _meta(True, season=s, direction="prev", k=1))

    for k in (2, 3):
        for tid, tmpl in KAFTER_TEMPLATES:
            for s in SEASONS:
                _item(items, "order_k", f"{tid}_k{k}", tmpl.format(Kw=KWORDS[k], S=s),
                      SEASONS, season_plus(s, k),
                      _meta(True, season=s, direction="after", k=k))
        for tid, tmpl in KBEFORE_TEMPLATES:
            for s in SEASONS:
                _item(items, "order_k", f"{tid}_k{k}", tmpl.format(Kw=KWORDS[k], S=s),
                      SEASONS, season_plus(s, -k),
                      _meta(True, season=s, direction="before", k=k))

    for start in FACT_CONVENTIONS:
        order = [season_plus(start, i) for i in range(4)]
        for pid, phrase in FACT_PHRASES:
            for pos in range(2, 5):
                ph = phrase.format(ORD=ORDINALS[pos], N=pos)
                prompt = f"If the year's seasons are listed starting with {start}, {ph}"
                _item(items, "fact_position", f"{start}_{pid}_p{pos}", prompt, SEASONS,
                      order[pos - 1], _meta(True, convention=start, position=pos))

    for tid, hemi, stated, tmpl in MONTH_SEASON_TEMPLATES:
        for m in MONTHS:
            _item(items, "month_to_season", tid, tmpl.format(M=m), SEASONS,
                  MONTH_SEASON[hemi][m],
                  _meta(True, hemisphere=hemi, hemisphere_stated=stated, month=m,
                        convention="meteorological"))

    for tid, hemi, part, tmpl in SEASON_MONTH_TEMPLATES:
        for s in SEASONS:
            m1, m2, m3 = SEASON_MONTHS[hemi][s]
            ans = {"first": m1, "middle": m2, "last": m3}[part]
            _item(items, "season_to_month", tid, tmpl.format(S=s, M1=m1, M2=m2), MONTHS, ans,
                  _meta(True, expect_season_concept=False, hemisphere=hemi,
                        hemisphere_stated=True, season=s, part=part,
                        convention="meteorological"))

    for tid, season, prompt in CHARACTERISTIC_ITEMS:
        _item(items, "characteristic", tid, prompt, SEASONS, season, _meta(True, season=season))

    for tid, tmpl, mode in CONTEXT_TEMPLATES:
        for s in SEASONS:
            ans = s if mode == "same" else season_plus(s, -1)
            _item(items, "usage_context", tid, tmpl.format(S=s), SEASONS, ans,
                  _meta(mode != "same", season=s, mode=mode))

    for eid, event, north_season in SOLSTICE_EVENTS:
        for hemi in ("northern", "southern"):
            ans = north_season if hemi == "northern" else season_plus(north_season, 2)
            prompt = (f"Using the astronomical convention, {event} marks the first day "
                      f"of a season, and in the {hemi} hemisphere that season is")
            _item(items, "solstice_equinox", f"{eid}_{hemi}", prompt, SEASONS, ans,
                  _meta(True, hemisphere=hemi, hemisphere_stated=True, event=eid))

    for tid, prompt, ans, kind, hemi in FALL_ITEMS:
        hemisphere = None if hemi == "northern_none" else hemi
        _item(items, "synonym_fall", tid, prompt, SEASONS if kind == "season" else MONTHS, ans,
              _meta(True, expect_season_concept=(kind == "season"),
                    hemisphere=hemisphere, hemisphere_stated=hemisphere is not None,
                    surface_form="fall", canonical="autumn"))

    for tid, kind, word, sense, prompt, ans, opts in POLYSEMY_ITEMS:
        _item(items, "polysemy", tid, prompt, opts, ans,
              _meta(not leaks_word(prompt, ans), expect_season_concept=False,
                    kind=kind, season_word=word, sense=sense))

    for i, (prompt, ans, opts) in enumerate(SANITY_ITEMS):
        _item(items, "sanity", f"s{i:02d}", prompt, opts, ans,
              _meta(True, expect_season_concept=False, seasonal=False))

    return items


def leaks_word(prompt, word):
    """True iff ``word`` appears in ``prompt`` as a whole word (case-insensitive)."""
    return re.search(rf"\b{re.escape(word)}\b", prompt, re.IGNORECASE) is not None


def leaks_answer(item):
    """True iff the answer appears in the prompt. Only meaningful for fill_in items;
    association items (usage_context 'same') legitimately name their season."""
    return leaks_word(item["prompt"], item["answer"])


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
    default_out = os.path.join(here, "evalsets", "seasons_v1.jsonl")
    ap = argparse.ArgumentParser(description="Generate the seasons_v1 eval set.")
    ap.add_argument("--out", default=default_out, help="output JSONL path")
    args = ap.parse_args()
    items = write_evalset(args.out)
    counts = category_counts(items)
    print(f"Wrote {len(items)} items -> {args.out}")
    for c in CATEGORIES:
        print(f"  {c:18s} {counts.get(c, 0)}")


if __name__ == "__main__":
    main()
