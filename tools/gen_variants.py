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
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from generator_support import emmy_executable, validate_package

ROOT = Path(__file__).resolve().parent.parent


def validate_or_reject(destination):
    try:
        validate_package(destination)
    except ValueError:
        rejected = ROOT / ".cache" / "rejected" / destination.name
        rejected.parent.mkdir(parents=True, exist_ok=True)
        if rejected.exists():
            raise ValueError(f"Invalid package retained at {destination}; rejected destination already exists") from None
        # Both paths are fixed children of the project and contain only generated fixtures.
        if not destination.resolve().is_relative_to((ROOT / "tests" / "generated").resolve()):
            raise ValueError("Refusing to move a package outside tests/generated") from None
        shutil.move(str(destination), str(rejected))
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hackathon", type=Path, default=ROOT.parent / "emmy-hackathon-2026", help="clone of the hackathon repo")
    ap.add_argument("--only", default=None, help="regenerate one variant id")
    ap.add_argument("--contains", default="", help="generate ids containing this text")
    ap.add_argument("--report", type=Path, help="write generation outcomes and exact errors as JSON")
    ap.add_argument("--resume", action="store_true", help="resume an unchanged spec list using --report checkpoints")
    args = ap.parse_args()
    generator = args.hackathon / "tools" / "generate.py"
    if not generator.exists():
        sys.exit(f"{generator} not found: pass --hackathon <clone of cloudrift-ai/emmy-hackathon-2026>")
    module_spec = importlib.util.spec_from_file_location("organiser_generator", generator)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    specs = module.load_specs(ROOT / "tests" / "variants.py")
    specs = [s for s in specs if args.contains in s["id"] and (not args.only or s["id"] == args.only)]
    if not specs:
        sys.exit("No matching variants")
    emmy = emmy_executable()
    rows = json.loads(args.report.read_text()) if args.resume and args.report and args.report.exists() else []
    completed = {r["id"] for r in rows}
    for row in rows:
        if row["ok"]:
            try:
                validate_or_reject(ROOT / "tests" / "generated" / row["id"])
            except ValueError as exc:
                row.update(ok=False, stage="package validation", error=str(exc))
    for spec in specs:
        if spec["id"] in completed:
            continue
        stage = "trace"
        with tempfile.TemporaryDirectory() as td:
            try:
                dump = Path(td) / "dump"
                module.trace(str(emmy), spec, dump, "sm_89")
                stage = "package"
                destination = ROOT / "tests" / "generated" / spec["id"]
                row = module.package(spec, dump, destination)
                stage = "package validation"
                validate_or_reject(destination)
            except Exception as exc:
                detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
                row = dict(id=spec["id"], ok=False, stage=stage, error=detail, code=spec["code"])
            rows.append(row)
            print(f"{'OK' if row['ok'] else 'FAIL'} {spec['id']}: "
                  + (str(row.get("kernels")) + " kernels" if row["ok"] else row["error"].strip().splitlines()[-1]), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    status = int(any(not r["ok"] for r in rows))
    # The reference CUDA is never used by our compiler; keep the suite lean.
    for ref in (ROOT / "tests" / "generated").glob("*/*.ref.cu"):
        ref.unlink()
    return status


if __name__ == "__main__":
    sys.exit(main())
