#!/usr/bin/env python3
"""Plain-assert sanity checks for Exp 1 (baseline) + shared launch infra.

Run from anywhere:  python runs/weekdays/test_exp1.py
No pytest required (but it also works under `pytest runs/weekdays/test_exp1.py`).

Checks:
  1. exp1_baseline.sh + pod_bootstrap.sh parse (`bash -n`); hf_push.py compiles.
  2. Every base_train flag exp1_baseline.sh passes actually exists in
     scripts/base_train.py's argparse (grep-level).
  3. base_eval flags used exist in scripts/base_eval.py.
  4. Token-budget math is self-consistent for d12 @ ratio 12:
     scaling params -> total_batch_size 2^19 -> 2520 steps -> ~1.321B tokens.
  5. Experiment identity strings are present (no VE, tag, HF repo, wandb run).
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NANO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP1 = os.path.join(HERE, "exp1_baseline.sh")
BOOTSTRAP = os.path.join(HERE, "pod_bootstrap.sh")
HF_PUSH = os.path.join(HERE, "hf_push.py")
BASE_TRAIN = os.path.join(NANO_ROOT, "scripts", "base_train.py")
BASE_EVAL = os.path.join(NANO_ROOT, "scripts", "base_eval.py")


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def test_files_exist():
    for p in (EXP1, BOOTSTRAP, HF_PUSH, BASE_TRAIN, BASE_EVAL):
        assert os.path.exists(p), f"missing: {p}"


def test_shell_scripts_parse():
    for p in (EXP1, BOOTSTRAP):
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        assert r.returncode == 0, f"bash -n failed for {p}:\n{r.stderr}"


def test_hf_push_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", HF_PUSH],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"py_compile failed for hf_push.py:\n{r.stderr}"


def _argparse_flags(path):
    """All --flag names declared via add_argument in a script."""
    src = _read(path)
    return set(re.findall(r'add_argument\(\s*["\'](--[a-z0-9\-]+)["\']', src))


def _flags_used(script_src):
    """--flags passed on command lines in a shell script (skip our own knobs)."""
    return set(re.findall(r'(?<![\w-])(--[a-z0-9\-]+)', script_src))


def test_base_train_flags_exist():
    exp1 = _read(EXP1)
    declared = _argparse_flags(BASE_TRAIN)
    # the base_train flags exp1_baseline.sh actually passes
    used = {
        "--depth", "--target-param-data-ratio", "--max-seq-len",
        "--device-batch-size", "--total-batch-size", "--num-iterations",
        "--model-tag", "--seed", "--no-value-embeds", "--fp8", "--fp8-recipe",
        "--save-every", "--save-optimizer", "--eval-every",
        "--core-metric-every", "--sample-every", "--run",
    }
    # sanity: those flags really are in the script text
    for f in ("--depth", "--no-value-embeds", "--target-param-data-ratio",
              "--seed", "--model-tag", "--run"):
        assert f in exp1, f"exp1_baseline.sh does not pass {f}"
    missing = used - declared
    assert not missing, f"flags used but not declared in base_train.py: {sorted(missing)}"


def test_base_eval_flags_exist():
    declared = _argparse_flags(BASE_EVAL)
    for f in ("--device-batch-size", "--model-tag"):
        assert f in declared, f"base_eval.py missing {f}"


def test_no_compact_tokens():
    # ignore comment lines (the flag is only referenced in a "NO --compact-tokens" note)
    code = "\n".join(ln for ln in _read(EXP1).splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "--compact-tokens" not in code, \
        "Exp 1 must use DEFAULT tokenization (no --compact-tokens passed)"


def test_identity_strings():
    s = _read(EXP1)
    assert "weekday-geometry-d${DEPTH}-baseline" in s or "weekday-geometry-d12-baseline" in s
    assert "kaushikreddyxyz/weekday-geometry-d12" in s
    assert "exp1-baseline" in s               # wandb run name
    assert "weekday-geometry" in s            # wandb project
    assert "--no-value-embeds" in s
    assert "baseline" in s                    # HF subdir


def test_budget_math():
    # d12 geometry, mirrors base_train.build_model_meta
    depth, aspect, head_dim, vocab, vpad = 12, 64, 128, 32768, 64
    md = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    assert md == 768, md
    pv = ((vocab + vpad - 1) // vpad) * vpad
    assert pv == 32768, pv
    scaling = depth * (4 * md * md + 2 * (md * 4 * md)) + pv * md
    assert scaling == 110_100_480, scaling

    ratio = 12                       # exp1_baseline.sh default RATIO
    total_batch = 2 ** 19            # d12 reference B_REF = 524,288
    assert total_batch == 524_288
    target_tokens = ratio * scaling
    steps = target_tokens // total_batch
    tokens = steps * total_batch
    assert steps == 2520, steps
    assert tokens == 1_321_205_760, tokens
    # within the "d12 default, < 2.5B ceiling, quick" band the brief asked for
    assert tokens < 2.5e9
    assert 1.0e9 < tokens < 1.5e9

    # the script's declared RATIO default must be 12 (shared horizon)
    m = re.search(r'RATIO=\$\{RATIO:-(\d+)\}', _read(EXP1))
    assert m and int(m.group(1)) == ratio, "exp1_baseline.sh RATIO default must be 12"


def test_cross_run_consistency():
    """ALL FOUR launch scripts pin the SAME shared knobs. These control the data
    order (nproc, shard count), the budget (iters/ratio), and comparability
    (seed, batch, seq len) — a divergence silently breaks the controlled study."""
    scripts = ["exp1_baseline.sh", "exp2_trainable.sh", "exp3_sphere.sh",
               "exp4_orthogonal.sh"]
    pins = {
        r'NPROC=\$\{NPROC:-(\d+)\}': "8",
        r'SEED=\$\{SEED:-(\d+)\}': "1337",
        r'DEPTH=\$\{DEPTH:-(\d+)\}': "12",
        r'RATIO=\$\{RATIO:-(\d+)\}': "12",
        r'NUM_ITERATIONS=\$\{NUM_ITERATIONS:-(\d+)\}': "2520",
        r'MAX_SEQ_LEN=\$\{MAX_SEQ_LEN:-(\d+)\}': "2048",
        r'DEVICE_BATCH_SIZE=\$\{DEVICE_BATCH_SIZE:-(\d+)\}': "32",
        r'TRAIN_SHARDS=\$\{TRAIN_SHARDS:-(\d+)\}': "45",
        r'PRECISION=\$\{PRECISION:-(\w+)\}': "bf16",
    }
    for s in scripts:
        src = _read(os.path.join(HERE, s))
        for pat, want in pins.items():
            m = re.search(pat, src)
            assert m and m.group(1) == want, \
                f"{s}: expected default {want} for {pat}, got {m.group(1) if m else 'MISSING'}"
        assert "--no-value-embeds" in src, f"{s}: missing --no-value-embeds"
        assert "--num-iterations" in src, f"{s}: horizon not explicitly pinned"
        assert 'nproc_per_node="$NPROC"' in src, f"{s}: torchrun must use $NPROC"
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert "--compact-tokens" not in code, f"{s}: must use default tokenization"


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
