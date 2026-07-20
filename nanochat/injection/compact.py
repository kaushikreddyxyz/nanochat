"""Compact-tokens mode (opt-in, OFF by default): a wrapper tokenizer whose
nanochat token boundaries never straddle gemma token boundaries.

Standard nanochat BPE can emit a token that crosses a gemma boundary (e.g. gemma
splits "XYZ ABC" as ["XYZ", " ABC"] but nanochat emits "Z A"), which forces the
alignment to pool across gemma tokens. Compact mode cuts the text at every gemma
span boundary and encodes each segment independently with the underlying
tokenizer, so every compact nanochat token nests inside exactly one gemma token
(gemma "XYZ ABC" -> "XY","Z"," ","A","BC"). Under compact tokenization the
overlap alignment is trivially 1:1, so all three pool policies coincide.

TWO LOUD CAVEATS (see nanochat/injection/README.md):
 (a) compact mode CHANGES the training token stream — breaking BPE merges at
     gemma boundaries inflates the token count, so a compact run is NOT
     bit-comparable to a standard-tokenizer baseline;
 (b) it needs gemma char-offsets in the hot loader path — already computed for
     alignment, so reuse them rather than gemma-tokenizing twice (see the
     wiring-gap note in the README).

Same interface as ``nanochat.tokenizer.RustBPETokenizer`` for the two consumers:
the ride-along loader (``encode``/``get_bos_token_id``) and the runtime probe
source (``.enc.encode_ordinary``/``.enc.decode_single_token_bytes``). Wire it in
via ``injection_train --compact-tokens``; core nanochat modules are untouched.
"""


def _cut_points(text, gemma_offsets):
    """Sorted unique char boundaries: 0, len(text), and every non-empty gemma
    span start/end. Segments between consecutive cuts never cross a gemma edge."""
    cuts = {0, len(text)}
    for s, e in gemma_offsets:
        if e > s:
            cuts.add(int(s))
            cuts.add(int(e))
    return sorted(c for c in cuts if 0 <= c <= len(text))


class _CompactEnc:
    """Inner-enc shim: ``encode_ordinary`` returns the compact id stream;
    ``decode_single_token_bytes`` delegates (ids are still base-vocab ids)."""

    def __init__(self, base_enc, gemma_encode):
        self._base = base_enc
        self._gemma_encode = gemma_encode

    def encode_ordinary(self, text):
        _, g_off = self._gemma_encode(text)
        cuts = _cut_points(text, g_off)
        ids = []
        for a, b in zip(cuts, cuts[1:]):
            seg = text[a:b]
            if seg:
                ids.extend(self._base.encode_ordinary(seg))
        return ids

    def decode_single_token_bytes(self, i):
        return self._base.decode_single_token_bytes(i)

    def __getattr__(self, name):
        return getattr(self._base, name)


class CompactGemmaTokenizer:
    """Wraps a nanochat tokenizer + a ``gemma_encode(text) -> (ids, offsets)``
    callable to produce a gemma-boundary-respecting token stream. Usable as both
    the loader's ``tokenizer`` and the source's ``nano_enc`` so both agree."""

    def __init__(self, base_tokenizer, gemma_encode):
        self._base = base_tokenizer
        self.enc = _CompactEnc(base_tokenizer.enc, gemma_encode)

    def get_bos_token_id(self):
        return self._base.get_bos_token_id()

    def encode_ordinary(self, text):
        return self.enc.encode_ordinary(text)

    def encode(self, text, prepend=None, append=None, num_threads=8):
        pre = None if prepend is None else (prepend if isinstance(prepend, int)
                                            else self._base.encode_special(prepend))
        app = None if append is None else (append if isinstance(append, int)
                                           else self._base.encode_special(append))

        def one(s):
            ids = self.enc.encode_ordinary(s)
            if pre is not None:
                ids.insert(0, pre)
            if app is not None:
                ids.append(app)
            return ids

        return one(text) if isinstance(text, str) else [one(s) for s in text]

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


def token_inflation(base_enc, gemma_encode, texts):
    """(#compact tokens / #standard tokens, n_standard, n_compact) over ``texts``
    — the measured stream-inflation number to quote when enabling compact mode."""
    compact = _CompactEnc(base_enc, gemma_encode)
    n_std = sum(len(base_enc.encode_ordinary(t)) for t in texts)
    n_cmp = sum(len(compact.encode_ordinary(t)) for t in texts)
    return (n_cmp / n_std if n_std else 1.0), n_std, n_cmp
