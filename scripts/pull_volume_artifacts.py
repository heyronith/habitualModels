#!/usr/bin/env python3
"""Pull JSONL artifacts from Modal volume into data/runs/<run_id>/."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--volume", default="habitual-checkpoints")
    args = ap.parse_args()
    dest = ROOT / "data" / "runs" / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    # modal volume get <name> <remote> <local>
    remote = f"runs/{args.run_id}"
    cmd = ["modal", "volume", "get", args.volume, remote, str(dest / "_vol")]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print("downloaded to", dest / "_vol")


if __name__ == "__main__":
    main()
