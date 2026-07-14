"""CPU test for compact-tokens mode (nanochat.injection.compact): the opt-in
wrapper tokenizer whose nanochat token boundaries never straddle gemma
boundaries. Uses a byte-pair fake base tokenizer that DOES straddle boundaries
under standard encoding, proving compact mode fixes it; measures the token-count
inflation (the number to quote in the README); and confirms that under compact
tokenization the runtime-probe overlap alignment is trivially 1:1 (mean == last).

Standalone: `python tests/test_compact_tokens.py`.
"""
import json
import os
import sys
import tempfile

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, REPO)

from nanochat.injection.compact import CompactGemmaTokenizer, token_inflation  # noqa: E402
from nanochat.injection.align import nanochat_char_offsets  # noqa: E402
from nanochat.injection.sources import RuntimeProbeScoreSource  # noqa: E402

fails = []


def check(cond, msg):
    print(f"  {'OK ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


class FakeBaseEnc:
    """Greedy 2-byte chunker (a BPE stand-in that merges across any boundary)."""

    def __init__(self):
        self._b2i, self._i2b, self._n = {}, {}, 1

    def _id(self, b):
        if b not in self._b2i:
            self._b2i[b] = self._n; self._i2b[self._n] = b; self._n += 1
        return self._b2i[b]

    def encode_ordinary(self, text):
        fb = text.encode("utf-8")
        return [self._id(fb[i:i + 2]) for i in range(0, len(fb), 2)]

    def decode_single_token_bytes(self, i):
        return self._i2b[i]

    def encode_special(self, s):
        return 0


class FakeBaseTok:
    def __init__(self):
        self.enc = FakeBaseEnc()

    def get_bos_token_id(self):
        return 0

    def encode_special(self, s):
        return 0


def word_gemma_encode(text):
    """Two-word gemma stand-in: 'XYZ ABC' -> spans (0,3),(3,7) (space rides the
    second word, like real gemma '▁ABC')."""
    import re
    ids, offs, i = [], [], 0
    for m in re.finditer(r"\s*\S+", text):
        ids.append(len(ids) + 1)
        offs.append((m.start(), m.end()))
        i += 1
    return ids, offs


TEXT = "XYZ ABC"
base = FakeBaseTok()
compact = CompactGemmaTokenizer(base, word_gemma_encode)

print("\n[A] compact tokens never straddle a gemma boundary")
_, g_off = word_gemma_encode(TEXT)
gem_spans = [(s, e) for s, e in g_off if e > s]

std_ids = base.enc.encode_ordinary(TEXT)
std_off = nanochat_char_offsets(base.enc, std_ids, TEXT)


def straddles(span):
    s, e = span
    if e <= s:
        return False
    return not any(gs <= s and e <= ge for gs, ge in gem_spans)  # not contained in a single gemma span


std_bad = [o for o in std_off if straddles(o)]
check(len(std_bad) > 0, f"standard tokenization DOES straddle a gemma boundary (offsets {std_bad}) — fixture is meaningful")

cmp_ids = compact.enc.encode_ordinary(TEXT)
cmp_off = nanochat_char_offsets(compact.enc, cmp_ids, TEXT)
check(all(not straddles(o) for o in cmp_off),
      f"every compact token nests inside one gemma span (offsets {cmp_off})")
# encode() batch path prepends bos and agrees with encode_ordinary
enc_batch = compact.encode([TEXT], prepend=compact.get_bos_token_id())
check(enc_batch[0][0] == 0 and enc_batch[0][1:] == cmp_ids, "encode(batch, prepend=bos) == [bos]+compact ids")

print("\n[B] token-count inflation (the number to quote)")
DOCS = ["XYZ ABC", "hello world foo", "aa bb cc dd ee", "single", "one two three four"]
ratio, n_std, n_cmp = token_inflation(base.enc, word_gemma_encode, DOCS)
print(f"  compact inflation over {len(DOCS)} fixture docs: {n_std} -> {n_cmp} tokens = {ratio:.3f}x")
check(ratio > 1.0, f"compact inflates the token count ({ratio:.3f}x > 1.0)")

print("\n[C] under compact tokenization the overlap alignment is trivially 1:1 (mean == last)")
K, LAYERS, CONCEPTS = 4, [6, 8, 14], ["a", "b", "c", "d"]
STORE = tempfile.mkdtemp(prefix="compact_probe_store_")
json.dump({"concepts": CONCEPTS, "layers": LAYERS, "families": {c: "fam" for c in CONCEPTS}},
          open(os.path.join(STORE, "columns.json"), "w"))
json.dump({"zero": [[0.0] * K for _ in LAYERS], "scale": [[1.0] * K for _ in LAYERS]},
          open(os.path.join(STORE, "quant.json"), "w"))
json.dump({"mean": [[0.0] * K for _ in LAYERS], "std": [[1.0] * K for _ in LAYERS]},
          open(os.path.join(STORE, "corpus_stats.json"), "w"))
n_gemma = len(word_gemma_encode(TEXT)[0])
sc = np.zeros((n_gemma, 3, K), np.int8)
for gi in range(n_gemma):
    sc[gi, 1, :] = gi + 1
np.save(os.path.join(STORE, "scores_00000.npy"), sc)
open(os.path.join(STORE, "docs_00000.jsonl"), "w").write(json.dumps({"doc": 0, "start": 0, "n": n_gemma}) + "\n")

n_tok = len(compact.enc.encode_ordinary(TEXT))
src_mean = RuntimeProbeScoreSource(STORE, shards=[0], layer=8, nano_enc=compact.enc,
                                   gemma_encode=word_gemma_encode, noise_sigma=0.0, align_policy="mean")
src_last = RuntimeProbeScoreSource(STORE, shards=[0], layer=8, nano_enc=compact.enc,
                                   gemma_encode=word_gemma_encode, noise_sigma=0.0, align_policy="last")
a_mean, _ = src_mean.lookup_by_row(0, 0, TEXT, n_tok)
a_last, _ = src_last.lookup_by_row(0, 0, TEXT, n_tok)
check(a_mean is not None and np.array_equal(a_mean, a_last),
      "compact tokenization => each nano token nests in one gemma => mean == last")
check(bool((a_mean != 0).any()), "compact-aligned activations are non-trivial (some tokens carry signal)")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_compact_tokens():
    assert not fails, f"{len(fails)} failures: {fails}"
