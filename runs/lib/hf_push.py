"""Best-effort mirror of a checkpoint dir to a HF model repo subfolder (never raises,
so it cannot kill a training run). One-shot, or --watch N loops every N seconds until
--stop-file appears (then one final push). HF_TOKEN via env/.env, never argv.
Usage: python runs/lib/hf_push.py --local-dir <dir> --path-in-repo <arm> [--repo id --watch 600 --stop-file f]
"""
import argparse
import os
import time

DEFAULT_REPO = "kaushikreddyxyz/nanochat-d12-injections"


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
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF model repo id")
    ap.add_argument("--local-dir", required=True, help="local checkpoint dir to mirror")
    ap.add_argument("--path-in-repo", required=True,
                    help="subfolder in the repo, e.g. seasons_sphere_L0")
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
