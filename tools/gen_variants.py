#!/usr/bin/env python3
"""Regenerate tests/generated/ from tests/variants.py.

    python tools/gen_variants.py --hackathon ../emmy-hackathon-2026 [--only p_exp]

Runs the organizers' own tools/generate.py (from a clone of the hackathon
repo) on our expression list, so each variant is traced by emmy and packaged
(IR, meta.json, expected.npz) exactly like a hidden-set problem. Needs torch
on top of emmy-ml; the compiler itself needs neither.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hackathon", type=Path, default=ROOT.parent / "emmy-hackathon-2026", help="clone of the hackathon repo")
    ap.add_argument("--only", default=None, help="regenerate one variant id")
    args = ap.parse_args()
    generator = args.hackathon / "tools" / "generate.py"
    if not generator.exists():
        sys.exit(f"{generator} not found: pass --hackathon <clone of cloudrift-ai/emmy-hackathon-2026>")
    cmd = [sys.executable, str(generator), str(ROOT / "tests" / "variants.py"), "--out", str(ROOT / "tests" / "generated")]
    if args.only:
        cmd += ["--only", args.only]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
