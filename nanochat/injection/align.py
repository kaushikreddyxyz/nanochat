"""Tokenizer bridge for cross-tokenizer alignment.

Vendored verbatim from the oracle-encodings superproject (stage-7 oracle
``align.py``, prefix mode validated at 7.08% gemma->qwen crossing) so that
``scripts/precompute_activations.py`` -- which aligns the frozen-encoder (qwen)
tokenization onto nanochat tokens -- is self-contained in this repo. The API
is frozen; if the upstream module changes, re-vendor deliberately.

Pure python/numpy. Tokenizer-agnostic: the alignment functions operate on
char-offset maps (lists of ``(start, end)`` int pairs, as returned by HF fast
tokenizers with ``return_offsets_mapping=True``), NOT on tokenizer objects.
This lets the same module bridge gemma->qwen today and qwen->nanochat later
-- only the offset maps change, not the code.

Terminology below follows the frozen naming in DESIGN.md: "gemma" = source
tokenizer/tokenization, "qwen" = target tokenizer/tokenization. Nothing in
the implementation actually depends on gemma or qwen specifically.

Modes
-----
``mode="prefix"`` (primary): for each source token, find the LAST target
token whose char span ends at or before the source token's end char (a
causal prefix-state lookup -- no future leakage). O(Tg + Tq) via a
two-pointer scan, since both offset sequences are char-monotonic
(non-decreasing end-char position) for any left-to-right tokenization.

``mode="boundary"`` (fallback, used when the prefix-mode crossing rate is
too high): re-tokenize each source token's substring independently with the
target tokenizer, forcing exact boundary alignment at the cost of losing
target-tokenizer context across the substring boundary. See
``gemma_to_qwen_map(..., mode="boundary")`` docstring for the calling
convention (it takes an extra ``retokenize`` callable, since re-tokenizing
requires an actual tokenizer -- this is the one place this module touches a
real tokenizer, always via a caller-supplied callable, never by importing
a tokenizer library itself).

Empty-span / special tokens: HF fast tokenizers report special tokens
(e.g. BOS/EOS) with an empty span, canonically ``(0, 0)`` but in general
any ``start == end``. This module treats ANY offset with ``start == end``
as an empty-span/special token:
  - on the SOURCE side: it is skipped (mapped to ``-1`` unconditionally,
    regardless of its position in the sequence).
  - on the TARGET side: it is excluded from consideration as an alignment
    anchor (it can never be "the last qwen token whose span ends <= ...").
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

Offset = Tuple[int, int]
Offsets = Sequence[Offset]


# --------------------------------------------------------------------------
# get_offsets: tokenizer -> (ids, offsets), normalized to the ORIGINAL string
# --------------------------------------------------------------------------

def get_offsets(
    tokenizer,
    text: str,
    add_special_tokens: bool = False,
) -> Tuple[List[int], List[Offset]]:
    """Tokenize ``text`` and return ``(ids, offsets)`` with offsets indexing
    into the ORIGINAL python string (character offsets, not byte offsets).

    Uses ``return_offsets_mapping=True`` on a fast tokenizer. Modern HF fast
    tokenizers (tokenizers >= 0.19-ish, transformers >= 4.x) already report
    proper char offsets into the original string for both byte-level BPE
    (GPT-2/Qwen family) and SentencePiece-based (Gemma/Llama family)
    tokenizers -- the historical "byte-level offset" quirk (offsets that
    index into a byte-remapped intermediate string, e.g. off-by-N around
    multibyte UTF-8 sequences, or endpoints that don't reconstruct the
    original substring) mostly affected older/slow tokenizers. This
    function still defensively validates and repairs offsets so callers
    never see offsets that violate the invariants downstream code depends
    on: ``0 <= start <= end <= len(text)`` and, for non-empty spans,
    ``end`` strictly increasing... actually non-decreasing across the
    sequence (standard left-to-right tokenization).

    Raises
    ------
    ValueError
        If ``tokenizer`` is not a fast tokenizer (offset mapping requires
        the Rust tokenizers backend) or offsets can't be repaired.
    """
    is_fast = getattr(tokenizer, "is_fast", False)
    if not is_fast:
        raise ValueError(
            "get_offsets requires a fast tokenizer (is_fast=True) so that "
            "return_offsets_mapping is supported; got "
            f"{type(tokenizer).__name__} with is_fast=False. Load with "
            "AutoTokenizer.from_pretrained(..., use_fast=True) (the "
            "default)."
        )

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    ids = list(enc["input_ids"])
    raw_offsets = list(enc["offset_mapping"])

    offsets = _normalize_offsets(text, raw_offsets)
    return ids, offsets


def _normalize_offsets(text: str, raw_offsets: Iterable[Offset]) -> List[Offset]:
    """Defensively clamp/repair a raw offset_mapping list against ``text``.

    Handles the known byte-level-BPE quirks: offsets that are inverted
    (start > end), offsets that exceed len(text) (can happen with certain
    normalization/pretokenization edge cases), and leaves empty spans
    (start == end, e.g. special tokens) untouched. Does NOT attempt to
    re-derive offsets from scratch (that would require the tokenizer's
    vocab) -- only clamps to valid, in-bounds, non-inverted ranges.
    """
    n = len(text)
    out: List[Offset] = []
    for start, end in raw_offsets:
        start = int(start)
        end = int(end)
        # Clamp to text bounds.
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        # Repair inversion defensively (shouldn't happen with modern fast
        # tokenizers, but a zero-cost guard against a corrupt offset).
        if end < start:
            start, end = end, start
        out.append((start, end))
    return out


# --------------------------------------------------------------------------
# nanochat_char_offsets: char spans for tiktoken/RustBPE ids (no fast-tokenizer
# offset mapping exists for tiktoken, so reconstruct from per-token byte lengths)
# --------------------------------------------------------------------------

_BYTE_LEN_LUTS = {}  # id(enc) -> (enc, byte-length lut int64[n], have bool[n])
#                       (enc kept alive so its id() can never be reused)


def _byte_len_lut(enc, ids_a):
    """Per-tokenizer cache of ``len(enc.decode_single_token_bytes(id))``: the
    per-token FFI loop this replaces was the single hottest GIL-held cost of the
    injection loader's production path (~0.5 ms/doc, see act_wait investigation
    2026-07-20). Grown lazily to the highest id seen; fills are idempotent, so
    concurrent lookup workers can only ever redo work, never corrupt it."""
    ent = _BYTE_LEN_LUTS.get(id(enc))
    if ent is None:
        ent = _BYTE_LEN_LUTS[id(enc)] = (enc, np.zeros(1024, np.int64),
                                         np.zeros(1024, bool))
    _, lut, have = ent
    hi = int(ids_a.max()) + 1 if ids_a.size else 0
    if hi > lut.size:
        n = max(hi, 2 * lut.size)
        lut = np.concatenate([lut, np.zeros(n - lut.size, np.int64)])
        have = np.concatenate([have, np.zeros(n - have.size, bool)])
        _BYTE_LEN_LUTS[id(enc)] = (enc, lut, have)
    miss = ids_a[~have[ids_a]]
    for i in np.unique(miss):
        lut[int(i)] = len(enc.decode_single_token_bytes(int(i)))
        have[int(i)] = True
    return lut


def nanochat_char_offsets(enc, ids, text):
    """Reconstruct (start, end) CHAR spans for tiktoken/RustBPE ``ids``. ``ids``
    must come from ``encode_ordinary(text)`` (no BOS): byte-level BPE partitions
    ``text.encode('utf-8')``, so per-token byte lengths must sum to the doc's
    byte length (asserted). A char belongs to the token holding its UTF-8 lead
    byte; pure-continuation-byte tokens get empty spans (align maps them to -1).

    Vectorized: token byte lengths come from a per-tokenizer LUT
    (``_byte_len_lut``) instead of a per-token FFI call, and the span list is
    assembled from numpy gathers. Output is identical to the original per-token
    implementation (validated element-wise on real ClimbMix docs)."""
    ids_a = np.asarray(ids, dtype=np.int64)
    lut = _byte_len_lut(enc, ids_a)
    b = np.zeros(ids_a.size + 1, dtype=np.int64)
    np.cumsum(lut[ids_a], out=b[1:])
    fb = text.encode("utf-8")
    assert int(b[-1]) == len(fb), (
        f"token byte lengths do not partition the document bytes "
        f"({int(b[-1])} != {len(fb)}); ids must come from encode_ordinary(text)")
    char_at = np.zeros(len(fb) + 1, dtype=np.int64)  # vectorized byte->char map
    if len(fb):
        fb_arr = np.frombuffer(fb, dtype=np.uint8)
        char_at[1:] = np.cumsum((fb_arr & 0xC0) != 0x80)
    return list(zip(char_at[b[:-1]].tolist(), char_at[b[1:]].tolist()))


# --------------------------------------------------------------------------
# prefix mode
# --------------------------------------------------------------------------

def _is_empty(span: Offset) -> bool:
    return span[1] <= span[0]


def _prefix_map(gemma_offsets: Offsets, qwen_offsets: Offsets) -> np.ndarray:
    """Two-pointer O(Tg + Tq) prefix-state alignment.

    Precondition (satisfied by any left-to-right tokenization, in
    particular anything produced by ``get_offsets``): both offset
    sequences have non-decreasing ``end`` char position as the index
    increases. Empty spans (start == end) are allowed anywhere and do not
    violate this precondition (their end just repeats/no-ops).
    """
    tg = len(gemma_offsets)

    # Filter target anchors: only non-empty target spans can be alignment
    # anchors, but we keep their ORIGINAL index (position in qwen_offsets)
    # since that's what the caller needs to index into e.g. qwen hidden
    # states, which include one row per raw qwen token position.
    qwen_anchors: List[Tuple[int, int]] = [
        (end, idx) for idx, (start, end) in enumerate(qwen_offsets) if end > start
    ]

    result = np.full(tg, -1, dtype=np.int64)
    j = 0
    n_anchors = len(qwen_anchors)
    last_valid = -1
    for i, (gs, ge) in enumerate(gemma_offsets):
        if ge <= gs:
            # Empty-span / special source token -> always -1.
            result[i] = -1
            continue
        while j < n_anchors and qwen_anchors[j][0] <= ge:
            last_valid = qwen_anchors[j][1]
            j += 1
        result[i] = last_valid
    return result


# --------------------------------------------------------------------------
# boundary mode
# --------------------------------------------------------------------------

RetokenizeFn = Callable[[str], int]


def _boundary_map(
    text: str,
    gemma_offsets: Offsets,
    retokenize: RetokenizeFn,
) -> List[List[int]]:
    """Boundary-constrained fallback.

    For each (non-empty-span) source token, re-tokenizes its substring
    ``text[start:end]`` independently with the target tokenizer via the
    caller-supplied ``retokenize`` callable, which must return the NUMBER
    of target subtokens produced (``retokenize(substr: str) -> int``) --
    e.g. ``lambda s: len(qwen_tok(s, add_special_tokens=False)["input_ids"])``.

    Returns a list of length ``len(gemma_offsets)``; entry ``i`` is a list
    of ints -- the LOCAL indices of source token ``i``'s target subtokens
    within a hypothetical flat "boundary sequence" formed by concatenating,
    in source-token order, the subtokens of every non-empty source token
    (empty-span source tokens contribute nothing and get ``[]``). A caller
    that wants the actual subtoken ids/hidden-states for that flat sequence
    re-invokes the SAME retokenization (this time collecting ids, not just
    counts) in the same order and concatenates -- ``align.py`` only computes
    the index structure, it never runs a tokenizer/model itself.

    This is O(Tg) calls to ``retokenize`` (each call itself does whatever
    work the caller's tokenizer needs); align.py's own bookkeeping is
    O(Tg + total subtokens).
    """
    result: List[List[int]] = []
    cursor = 0
    for start, end in gemma_offsets:
        if end <= start:
            result.append([])
            continue
        substr = text[start:end]
        n = int(retokenize(substr))
        if n < 0:
            raise ValueError(
                f"retokenize({substr!r}) returned negative count {n}"
            )
        result.append(list(range(cursor, cursor + n)))
        cursor += n
    return result


# --------------------------------------------------------------------------
# public API (frozen, see DESIGN.md)
# --------------------------------------------------------------------------

def gemma_to_qwen_map(
    text: str,
    gemma_offsets: Offsets,
    qwen_offsets: Offsets,
    mode: str = "prefix",
    retokenize: RetokenizeFn | None = None,
):
    """Align source ("gemma") tokens to target ("qwen") tokens.

    Parameters
    ----------
    text: the raw string both tokenizations were computed over.
    gemma_offsets: [Tg] list/array of ``(start, end)`` char spans (source).
    qwen_offsets: [Tq] list/array of ``(start, end)`` char spans (target).
        Unused when ``mode="boundary"`` (may be passed as ``None`` /
        ``[]`` in that case) -- boundary mode re-tokenizes per-substring
        instead of using a whole-text target tokenization.
    mode: "prefix" (default) or "boundary".
    retokenize: required iff ``mode="boundary"``; see ``_boundary_map``.

    Returns
    -------
    mode="prefix": ``np.ndarray`` of shape ``[Tg]``, dtype int64. Entry
        ``i`` is the index (into ``qwen_offsets``) of the LAST target token
        whose span ends at or before source token ``i``'s end char, or
        ``-1`` if no such target token exists yet (including: source token
        ``i`` itself has an empty span). O(Tg + Tq).
    mode="boundary": ``List[List[int]]`` of length ``Tg``; see
        ``_boundary_map`` docstring for the index semantics.
    """
    if mode == "prefix":
        return _prefix_map(gemma_offsets, qwen_offsets)
    elif mode == "boundary":
        if retokenize is None:
            raise ValueError(
                "mode='boundary' requires a retokenize callable: "
                "retokenize(substr: str) -> int (number of target "
                "subtokens for that substring)."
            )
        return _boundary_map(text, gemma_offsets, retokenize)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'prefix' or 'boundary'")


def crossing_rate(text: str, gemma_offsets: Offsets, qwen_offsets: Offsets) -> float:
    """Fraction of (non-empty-span) source tokens whose end char is NOT
    exactly the end char of some target token -- i.e. positions where the
    prefix-mode alignment (``mode="prefix"``) necessarily lags behind the
    true source-token boundary because no target token ends there exactly.

    Empty-span source tokens (special tokens) are excluded from both the
    numerator and denominator: they always map to -1 by construction, so
    they're not a meaningful "crossing" and would just dilute the rate.

    Returns 0.0 for a text with no non-empty source tokens (e.g. empty
    text, or a source sequence of only special tokens).
    """
    qwen_ends = {end for start, end in qwen_offsets if end > start}
    n_total = 0
    n_cross = 0
    for gs, ge in gemma_offsets:
        if ge <= gs:
            continue
        n_total += 1
        if ge not in qwen_ends:
            n_cross += 1
    if n_total == 0:
        return 0.0
    return n_cross / n_total
