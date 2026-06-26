"""Oracle plumbing smoke test (CPU) — geometric-manifold injection in nanochat.

The nanochat-side analogue of ``modular_addition/oracle/experiments/exp00_smoke``:
validates the whole oracle path end-to-end on a laptop (fp32, CPU, tiny model,
no wandb / no data downloads). NOT a real pretraining run.

Sub-tests
  1. reproducibility — same seed builds the same model.
  2. geometry — ring/sphere coords are unit-norm & evenly spaced; the table is
     ``amp`` on the chosen ids, exactly zero elsewhere.
  3. inject gate / ablation identity — the oracle changes the forward when on;
     ``inject=False`` is bit-identical to having no oracle at all.
  4. frozen oracle — the table is not a Parameter, takes no grad, and is
     unchanged by an optimizer step.
  5. end-to-end uptake — on a ring-rotation task with the token-identity
     embeddings (wte + value_embeds) zeroed and frozen, the oracle is the ONLY
     per-token signal; training drives loss down and ablating the oracle at eval
     spikes cross-entropy back to ~uniform (large ΔCE) — i.e. the model learned
     to USE the injected geometry.

Run:  cd nanochat && python -m nanochat.oracle.smoke
"""
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.oracle import inject


# --------------------------------------------------------------------------- #
# Tiny-model + synthetic-task helpers
# --------------------------------------------------------------------------- #
N_RING = 12          # ring tokens (a nod to the months-on-a-circle example)
SEQ_LEN = 64         # rotary/context budget; sequences fed are shorter
T = 16               # task sequence length
DEVICE = "cpu"


def build_tiny_model(vocab_size, *, seed=0, n_layer=2, n_embd=64, n_head=2,
                     device=DEVICE):
    """Materialize a tiny GPT the same way base_train does (meta -> to_empty ->
    init_weights), seeded for reproducibility. window_pattern='L' keeps full
    context so the SDPA fallback takes the simple causal path on CPU."""
    cfg = GPTConfig(sequence_len=SEQ_LEN, vocab_size=vocab_size, n_layer=n_layer,
                    n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
                    window_pattern="L")
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    torch.manual_seed(seed)                  # deterministic init regardless of meta-build RNG
    model.init_weights()
    return model


def ring_batch(batch, *, n=N_RING, step=1, seq=T, generator=None):
    """A batch of ring rotations: s_t = (s_0 + step*t) mod n. Returns (idx,
    targets), each (batch, seq), with targets = idx shifted by one (next token)."""
    s0 = torch.randint(0, n, (batch, 1), generator=generator)
    steps = torch.arange(seq + 1)
    full = (s0 + step * steps) % n           # (batch, seq+1)
    return full[:, :-1].contiguous(), full[:, 1:].contiguous()


def silence_token_identity(model):
    """Zero + freeze every per-token-id signal except the oracle, so the oracle
    is provably the only thing that distinguishes tokens: the token embedding
    (wte) and the value embeddings (value_embeds). lm_head and the trunk stay
    trainable. Returns the list of frozen parameter names."""
    with torch.no_grad():
        model.transformer.wte.weight.zero_()
        for ve in model.value_embeds.values():
            ve.weight.zero_()
    return inject.freeze_params(model, ["wte", "value_embeds"])


@torch.no_grad()
def eval_loss(model, idx, targets, *, inject_on):
    was = getattr(model, "inject", True)
    model.inject = inject_on
    loss = model(idx, targets=targets).item()
    model.inject = was
    return loss


# --------------------------------------------------------------------------- #
# Sub-tests
# --------------------------------------------------------------------------- #
def test_reproducibility():
    a = build_tiny_model(N_RING, seed=0)
    b = build_tiny_model(N_RING, seed=0)
    assert torch.allclose(a.transformer.wte.weight, b.transformer.wte.weight), \
        "same seed must build the same model"
    idx, tgt = ring_batch(8, generator=torch.Generator().manual_seed(1))
    assert abs(a(idx, targets=tgt).item() - b(idx, targets=tgt).item()) < 1e-9
    print("[repro] OK")


def test_geometry():
    rc = inject.ring_coords(N_RING)
    assert rc.shape == (N_RING, 2)
    assert torch.allclose(rc.norm(dim=1), torch.ones(N_RING), atol=1e-6), \
        "ring points must be unit-norm"
    # consecutive angular gap is constant 2*pi/N
    gaps = torch.atan2(rc[:, 1], rc[:, 0]).diff() % (2 * torch.pi)
    assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-5), \
        "ring points must be evenly spaced"
    sc = inject.sphere_coords(20)
    assert torch.allclose(sc.norm(dim=1), torch.ones(20), atol=1e-5), \
        "sphere points must be unit-norm"

    n_embd, amp = 64, 0.7
    ids = list(range(N_RING))
    orc = inject.make_ring_oracle(N_RING, n_embd, ids, dims=(0, 1), amp=amp)
    tbl = orc.table
    assert tbl.shape == (N_RING, n_embd)
    # exactly amp on the chosen ids, exactly zero everywhere else (incl. other dims)
    assert torch.allclose(tbl[:, :2].norm(dim=1), torch.full((N_RING,), amp), atol=1e-6)
    assert tbl[:, 2:].abs().max().item() == 0.0, "oracle must not touch other dims"
    print(f"[geometry] OK  (ring unit-norm, evenly spaced; table norm == amp={amp})")


def test_inject_gate():
    model = build_tiny_model(N_RING, seed=2)
    orc = inject.make_ring_oracle(N_RING, model.config.n_embd, list(range(N_RING)),
                                  amp=1.0)
    inject.attach_oracle(model, orc)
    idx, _ = ring_batch(8, generator=torch.Generator().manual_seed(3))

    model.inject = True
    on = model(idx)                                   # logits (B, T, vocab)
    model.inject = False
    off = model(idx)
    assert (on - off).abs().max().item() > 1e-3, "oracle must change the forward"

    # inject=False must be the exact same code path as having no oracle at all
    inject.detach_oracle(model)
    none = model(idx)
    assert torch.equal(off, none), "inject=False must equal the no-oracle forward"
    print(f"[gate] OK  ||on-off||_max={ (on - off).abs().max().item():.4f}")


def test_frozen():
    model = build_tiny_model(N_RING, seed=4)
    orc = inject.make_ring_oracle(N_RING, model.config.n_embd, list(range(N_RING)),
                                  amp=1.0)
    inject.attach_oracle(model, orc)
    # not a Parameter, requires no grad
    param_ids = {id(p) for p in model.parameters()}
    assert id(orc.table) not in param_ids, "oracle table must not be a Parameter"
    assert not orc.table.requires_grad
    before = orc.table.clone()
    idx, tgt = ring_batch(16, generator=torch.Generator().manual_seed(5))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model(idx, targets=tgt).backward()
    opt.step()
    assert orc.table.grad is None, "no gradient should reach the oracle table"
    assert torch.equal(orc.table, before), "oracle table must be unchanged by a step"
    print("[frozen] OK  (no grad, not a Parameter, unchanged by optimizer step)")


def test_end_to_end_uptake(steps=600, lr=1e-2, batch=128):
    """Train on the ring-rotation task with the oracle as the only token signal;
    show loss drops and ablating the oracle spikes CE (the model uses it)."""
    model = build_tiny_model(N_RING, seed=6)
    frozen = silence_token_identity(model)
    assert any("wte" in f for f in frozen) and any("value_embeds" in f for f in frozen)
    orc = inject.make_ring_oracle(N_RING, model.config.n_embd, list(range(N_RING)),
                                  amp=1.0)
    inject.attach_oracle(model, orc)

    gen = torch.Generator().manual_seed(7)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    eval_idx, eval_tgt = ring_batch(256, generator=torch.Generator().manual_seed(99))
    init_loss = eval_loss(model, eval_idx, eval_tgt, inject_on=True)

    model.train()
    for _ in range(steps):
        idx, tgt = ring_batch(batch, generator=gen)
        loss = model(idx, targets=tgt)
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()

    loss_on = eval_loss(model, eval_idx, eval_tgt, inject_on=True)
    loss_off = eval_loss(model, eval_idx, eval_tgt, inject_on=False)
    uniform = torch.log(torch.tensor(float(N_RING))).item()   # ln(12) ≈ 2.485
    delta = loss_off - loss_on
    print(f"[uptake] init={init_loss:.3f}  trained_on={loss_on:.3f}  "
          f"ablated_off={loss_off:.3f}  ΔCE={delta:+.3f}  (uniform≈{uniform:.3f})")
    assert loss_on < 1.0, f"model failed to learn the ring task (loss_on={loss_on:.3f})"
    assert delta > 1.0, f"ablation should spike CE; ΔCE={delta:.3f} too small"
    # frozen wte stayed exactly zero -> token identity really did come only via the oracle
    assert model.transformer.wte.weight.abs().max().item() == 0.0
    print("[uptake] OK  (learned to use the injected ring; ablation is causal)")


def main():
    torch.use_deterministic_algorithms(False)   # SDPA fallback has non-deterministic kernels; fine for a smoke test
    print(f"oracle smoke test — device={DEVICE}, N_RING={N_RING}, T={T}")
    test_reproducibility()
    test_geometry()
    test_inject_gate()
    test_frozen()
    test_end_to_end_uptake()
    print("\n✅ smoke test passed")


if __name__ == "__main__":
    main()
