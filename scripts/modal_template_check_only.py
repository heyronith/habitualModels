#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modal_app import app, template_check_qwen


def main() -> None:
    print("starting", flush=True)
    with app.run():
        print("calling remote", flush=True)
        r = template_check_qwen.remote("qwen3_1_7b")
    out = ROOT / "reports" / "T1_template_check_qwen3_1_7b.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    print("all_single", r["all_single"], flush=True)
    print("equal", r["equal_atol_1e-5"], flush=True)
    print("gpu_seconds", r["gpu_seconds"], flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
