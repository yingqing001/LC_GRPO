"""Download the video-generation checkpoints listed in the README.

Usage:
    python download.py              # download all checkpoints
    python download.py hunyuan      # download a subset by key (space/comma separated)
    python download.py --list       # list available checkpoints

Requires `huggingface_hub` (installed by env_setup.sh).
"""

import argparse
import sys

from huggingface_hub import snapshot_download

# key -> (repo_id, local_dir) as specified in the README "video generation" section.
CHECKPOINTS = {
    "hunyuan": ("hunyuanvideo-community/HunyuanVideo", "./data/HunyuanVideo"),
    "skyreels": ("xzyhku/SkyReels-V1-I2V", "./data/SkyReels-I2V"),
    "qwen2vl": ("Qwen/Qwen2-VL-2B-Instruct", "./Qwen2-VL-2B-Instruct"),
    "videoalign": ("KwaiVGI/VideoReward", "./videoalign_ckpt"),
}


def download(key):
    repo_id, local_dir = CHECKPOINTS[key]
    print(f"[{key}] downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[{key}] done")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*",
                        help="checkpoints to download (default: all). "
                             f"Available: {', '.join(CHECKPOINTS)}")
    parser.add_argument("--list", action="store_true", help="list available checkpoints and exit")
    args = parser.parse_args()

    if args.list:
        for key, (repo_id, local_dir) in CHECKPOINTS.items():
            print(f"{key:12s} {repo_id} -> {local_dir}")
        return

    # allow comma-separated keys too, e.g. "hunyuan,skyreels"
    keys = []
    for k in args.keys:
        keys.extend(part for part in k.split(",") if part)
    keys = keys or list(CHECKPOINTS)

    unknown = [k for k in keys if k not in CHECKPOINTS]
    if unknown:
        sys.exit(f"Unknown checkpoint(s): {', '.join(unknown)}. "
                 f"Available: {', '.join(CHECKPOINTS)}")

    for key in keys:
        download(key)


if __name__ == "__main__":
    main()
