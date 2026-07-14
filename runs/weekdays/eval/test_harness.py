"""CPU tests for the weekday-geometry eval harness. NO network / rustbpe / gemma:
a byte-greedy fake nano tokenizer + synthetic gemma z stand in for the injected
dependencies. Plain asserts (run: `python runs/weekdays/eval/test_harness.py`).

Covers the FOUNDATION invariants the sibling suites build on:
  * gate_scale=0 forward is BIT-IDENTICAL to the no-site (acts=None) forward on a
    tiny d=64 model that HAS a site (gate 0 = exact framework no-op);
  * the gate is restored exactly after the gate_scale context manager;
  * build_acts threshold zeroes sub-threshold rows EXACTLY;
  * build_acts reuses training alignment semantics (overlap broadcast + mean over
    a multi-gemma span), verified on a hand-checked case;
  * ce_report buckets (injected / after / other) partition valid positions with
    the right counts and means.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import harness  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEnc:
    """Byte-greedy tokenizer mimicking the tiktoken surface build_acts needs:
    encode_ordinary (longest-match over a fixed byte vocab) +
    decode_single_token_bytes. Token byte-lengths partition the text bytes, so
    align.nanochat_char_offsets works unchanged."""

    def __init__(self, vocab):
        self.vocab = [v if isinstance(v, bytes) else v.encode("utf-8") for v in vocab]
        self.by_bytes = {v: i for i, v in enumerate(self.vocab)}
        self.maxlen = max(len(v) for v in self.vocab)

    def encode_ordinary(self, text):
        b = text.encode("utf-8")
        ids, i = [], 0
        while i < len(b):
            for L in range(min(self.maxlen, len(b) - i), 0, -1):
                if b[i:i + L] in self.by_bytes:
                    ids.append(self.by_bytes[b[i:i + L]])
                    i += L
                    break
            else:
                raise ValueError(f"FakeEnc: no token at byte {i} of {text!r}")
        return ids

    def decode_single_token_bytes(self, i):
        return self.vocab[int(i)]


def _tiny_gpt_with_site():
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.injection.sites import InjectionCfg
    cfg = GPTConfig(sequence_len=64, vocab_size=64, n_layer=4, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    torch.manual_seed(0)
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu")
    torch.manual_seed(0)
    m.init_weights()
    m.eval()
    m.setup_injection_sites([InjectionCfg(name="weekdays", r=7, after_block=1, gate=0.05)])
    return m


# --------------------------------------------------------------------------- #
# 1. gate_scale=0 == no-site forward (exact) + 2. gate restored
# --------------------------------------------------------------------------- #
def test_gate_scale_zero_is_exact_noop_and_restores():
    m = _tiny_gpt_with_site()
    torch.manual_seed(1)
    ids = torch.randint(0, 60, (2, 16))
    acts = np.random.RandomState(0).randn(2, 16, 7).astype(np.float32)  # dense nonzero

    off = harness.forward_metrics(m, ids, acts=acts, gate_scale=0.0)["logits"]
    vanilla = harness.forward_metrics(m, ids, acts=None)["logits"]
    assert torch.equal(off, vanilla), "gate_scale=0 must be bit-identical to the no-site forward"

    # gate must be restored exactly (context manager) after the scaled forward.
    g = m.injection_sites["weekdays"].gate.detach().clone()
    assert torch.equal(g, torch.tensor(0.05)), "gate not restored after gate_scale=0"
    harness.forward_metrics(m, ids, acts=acts, gate_scale=0.5)
    assert torch.equal(m.injection_sites["weekdays"].gate.detach(), torch.tensor(0.05)), \
        "gate not restored after gate_scale=0.5"

    # gate_scale=1 with real acts DOES change the output (site is live).
    on = harness.forward_metrics(m, ids, acts=acts, gate_scale=1.0)["logits"]
    assert not torch.equal(on, vanilla), "gate_scale=1 with nonzero acts must change the logits"


# --------------------------------------------------------------------------- #
# 3 + 4. build_acts alignment (broadcast + mean) and threshold zeroing
# --------------------------------------------------------------------------- #
def _align_fixture():
    # text: gemma tokens "monday"(0,6)=g0, " "(6,7)=g1, "tue"(7,10)=g2;
    # nano tokens "mon"(0,3), "day"(3,6), " tue"(6,10).
    #   nano0,nano1 nest inside gemma "monday" -> both BROADCAST g0
    #   nano2 " tue" spans gemma " " + "tue"   -> MEAN(g1, g2)
    text = "monday tue"
    enc = FakeEnc(["mon", "day", " tue"])
    nano_ids = enc.encode_ordinary(text)
    assert len(nano_ids) == 3
    offsets = np.array([[0, 6], [6, 7], [7, 10]], np.int64)      # gemma char spans
    g0 = np.array([3, 0, 0, 0, 0, 0, 0], np.float32)            # max 3 >= 2
    g1 = np.array([1, 0, 0, 0, 0, 0, 0], np.float32)
    g2 = np.array([0, 1, 0, 0, 0, 0, 0], np.float32)
    gemma_z = np.stack([g0, g1, g2])
    return text, enc, nano_ids, offsets, gemma_z, (g0, g1, g2)


def test_build_acts_alignment_broadcast_and_mean():
    text, enc, nano_ids, offsets, gemma_z, (g0, g1, g2) = _align_fixture()
    # threshold off (-inf) so we can inspect the raw pooled values.
    out = harness.build_acts(text, nano_ids, gemma_z, offsets, threshold=-1e9,
                             policy="mean", nano_enc=enc)
    assert out.shape == (3, 7)
    assert np.array_equal(out[0], g0), "nano0 must broadcast gemma 'monday'"
    assert np.array_equal(out[1], g0), "nano1 must broadcast gemma 'monday'"
    assert np.allclose(out[2], (g1 + g2) / 2.0), "nano2 must be the MEAN over covering gemma tokens"

    # 'last' policy keeps the rightmost covering gemma token for the multi span.
    out_last = harness.build_acts(text, nano_ids, gemma_z, offsets, threshold=-1e9,
                                  policy="last", nano_enc=enc)
    assert np.array_equal(out_last[2], g2), "'last' policy must keep the rightmost covering gemma token"


def test_build_acts_threshold_exact_zero():
    text, enc, nano_ids, offsets, gemma_z, (g0, g1, g2) = _align_fixture()
    out = harness.build_acts(text, nano_ids, gemma_z, offsets, threshold=2.0,
                             policy="mean", nano_enc=enc)
    # nano0,nano1 = g0 (max 3 >= 2) survive; nano2 = mean(g1,g2)=[.5,.5,..] (max .5 < 2) zeroed.
    assert np.array_equal(out[0], g0)
    assert np.array_equal(out[1], g0)
    assert np.count_nonzero(out[2]) == 0, "sub-threshold row must be EXACT zero"
    assert out[2].sum() == 0.0


# --------------------------------------------------------------------------- #
# 5. ce_report bucket counts + means
# --------------------------------------------------------------------------- #
def test_ce_report_buckets():
    # positions 0..5; pos0 NaN (no context). Inject input positions 2 and 4.
    ptc = np.array([[np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]], np.float32)
    acts = np.zeros((1, 6, 7), np.float32)
    acts[0, 2, 0] = 1.0     # injected token at pos 2
    acts[0, 4, 3] = 2.0     # injected token at pos 4
    rep = harness.ce_report(ptc, acts)

    assert rep["n_overall"] == 5 and abs(rep["ce_overall"] - 3.0) < 1e-6   # mean(1..5)
    assert rep["n_injected"] == 2 and abs(rep["ce_injected"] - 3.0) < 1e-6  # mean(2,4)
    assert rep["n_after_injected"] == 2 and abs(rep["ce_after_injected"] - 4.0) < 1e-6  # mean(3,5)
    assert rep["n_other"] == 1 and abs(rep["ce_other"] - 1.0) < 1e-6        # pos1
    # buckets partition the valid positions
    assert rep["n_injected"] + rep["n_after_injected"] + rep["n_other"] == rep["n_overall"]


# --------------------------------------------------------------------------- #
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nAll {len(tests)} harness tests passed.")


if __name__ == "__main__":
    main()
