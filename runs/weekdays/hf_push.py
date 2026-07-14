#!/usr/bin/env python3
"""Push a nanochat checkpoint dir (+ report) to a HF model repo as artifacts complete.

Mirrors the oracles/ idiom: HfApi().create_repo(exist_ok=True) + upload_folder into
a <path-in-repo>/ subdir. Best-effort — every push is wrapped so it can NEVER raise
and kill a training run. HF_TOKEN is read natively by huggingface_hub (via env or
nanochat/.env); it is never passed on argv.

Shared infra: the 4 weekday-geometry runs can all reuse this (change --path-in-repo).

Usage:
  one-shot:  python hf_push.py --repo kaushikreddyxyz/weekday-geometry-d12 \
                 --local-dir <ckpt_dir> --path-in-repo baseline
  watch:     python hf_push.py ... --watch 600 --stop-file /path/to/.pushdone
             (loop: push every 600s; when --stop-file exists do one final push, exit)
"""
import argparse
import os
import time


def _push_once(api, repo, local_dir, path_in_repo):
    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        print(f"[hf_push] nothing to push yet: {local_dir}", flush=True)
        return
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
    )
    print(f"[hf_push] pushed {local_dir} -> {repo}/{path_in_repo}", flush=True)


def _attempt(api, repo, local_dir, path_in_repo):
    try:
        _push_once(api, repo, local_dir, path_in_repo)
    except Exception as e:  # noqa: BLE001 — push is best-effort, never kills training
        print(f"[hf_push] FAILED (non-fatal): {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF model repo id, e.g. kaushikreddyxyz/weekday-geometry-d12")
    ap.add_argument("--local-dir", required=True, help="local checkpoint dir to mirror")
    ap.add_argument("--path-in-repo", default="baseline", help="subfolder in the repo")
    ap.add_argument("--watch", type=int, default=0, help="seconds between pushes; 0 = one-shot")
    ap.add_argument("--stop-file", default="", help="when this file exists, do a final push and exit")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    if args.watch <= 0:
        _attempt(api, args.repo, args.local_dir, args.path_in_repo)
        return

    while True:
        _attempt(api, args.repo, args.local_dir, args.path_in_repo)
        if args.stop_file and os.path.exists(args.stop_file):
            _attempt(api, args.repo, args.local_dir, args.path_in_repo)  # final sync
            print("[hf_push] stop-file seen; final push done; exiting", flush=True)
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
