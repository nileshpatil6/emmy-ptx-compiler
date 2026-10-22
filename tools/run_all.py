#!/usr/bin/env python3
"""Compile and test every problem with our compiler, on the remote runner.

    python tools/run_all.py                    # open set + generated variants
    python tools/run_all.py tests/open         # one suite
    python tools/run_all.py -k r0              # only problems whose id contains "r0"

For each problem folder (meta.json + *.ir.json + expected.npz) this does what
the hackathon's submit.py does: run ``python compile.py <ir>`` for each
kernel, assemble and launch it under meta.json's geometry, chain
intermediate buffers between kernels, and compare the final outputs with
expected.npz using the problem's rtol / atol. It then prints a pass table per
category (the first letter of the problem id: p, r, m, f, a).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Runner, RunnerError  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUITES = [ROOT / "tests" / "open", ROOT / "tests" / "generated", ROOT / "tests" / "synthetic"]
SUITE_NAMES = ("open", "generated", "synthetic")
CATEGORIES = {"p": "pointwise", "r": "reductions", "m": "matmul", "f": "fused", "a": "attention"}


def run_problem(pdir: Path, runner: Runner, timing: bool) -> dict:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz")) if (pdir / "expected.npz").exists() else {}
    produced: dict[str, list] = {}
    result = {"id": pdir.name, "suite": pdir.parent.name, "ok": True, "detail": "", "us": 0.0, "compile_s": 0.0}
    for k in meta["kernels"]:
        ir = pdir / f"{k['entry']}.ir.json"
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
        result["compile_s"] += time.perf_counter() - t0
        if proc.returncode != 0:
            return {**result, "ok": False, "detail": f"compile.py failed: {proc.stderr.strip()[-300:]}"}
        c = runner.compile(proc.stdout)
        if not c.get("ok"):
            return {**result, "ok": False, "detail": "ptxas: " + c.get("stderr", "")[:600]}
        if k["entry"] not in c.get("entry_points", []):
            return {**result, "ok": False, "detail": f"entry points {c.get('entry_points')}"}
        args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        r = runner.run(
            {
                "cubin_id": c["cubin_id"],
                "entry": k["entry"],
                "grid": k["grid"],
                "block": k["block"],
                "smem_bytes": k["smem_bytes"],
                "zero_outputs": k["zero_outputs"],
                "args": args,
                "warmup": 3 if timing else 0,
                "iters": 20 if timing else 1,
                "max_return": max(4096, need),
            }
        )
        if not r.get("ok"):
            return {**result, "ok": False, "detail": f"launch failed ({r.get('stage')}): {r.get('stderr', '')[:300]}"}
        result["us"] += float(r.get("median_us", 0.0))
        result["registers"] = c.get("registers", {}).get(k["entry"])
        for name, out in r["outputs"].items():
            produced[name] = out["values"]
            if name in expected:
                ref = expected[name]
                got = np.array(out["values"], dtype=np.float32).reshape(ref.shape)
                if not np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"]):
                    err = float(np.nanmax(np.abs(got - ref)))
                    result["ok"] = False
                    result["detail"] += f"{name}: max abs err {err:.3e} (rtol={meta['rtol']} atol={meta['atol']}) "
    return result


def markdown_table(results: list[dict]) -> str:
    lines = ["| category | open set | generated variants | synthetic IR |", "|---|---|---|---|"]
    for letter, name in CATEGORIES.items():
        cells = []
        for suite in SUITE_NAMES:
            rs = [r for r in results if r["suite"] == suite and r["id"].startswith(letter)]
            cells.append(f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "| suite | problem | result | median us | registers |", "|---|---|---|---|---|"]
    for r in results:
        status = "PASS" if r["ok"] else "FAIL " + r["detail"]
        lines.append(f"| {r['suite']} | {r['id']} | {status} | {r['us']:.1f} | {r.get('registers', '')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="*", type=Path, help="folders of problems (default: tests/open tests/generated)")
    ap.add_argument("-k", default="", help="only problems whose id contains this text")
    ap.add_argument("--timing", action="store_true", help="20 timed launches per kernel (median reported)")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached runner answers")
    ap.add_argument("--json", type=Path, help="write per-problem results here")
    ap.add_argument("--markdown", type=Path, help="write a results table (RESULTS.md style) here")
    args = ap.parse_args()

    runner = Runner(use_cache=not args.no_cache)
    problems = sorted(
        (p for s in (args.suites or SUITES) if s.exists() for p in s.iterdir() if (p / "meta.json").exists() and args.k in p.name),
        key=lambda p: (SUITE_NAMES.index(p.parent.name) if p.parent.name in SUITE_NAMES else 9, p.name),
    )
    results = []
    for p in problems:
        try:
            res = run_problem(p, runner, args.timing)
        except RunnerError as exc:
            print(f"runner error: {exc}")
            return 2
        results.append(res)
        status = "PASS" if res["ok"] else "FAIL"
        timing = f"{res['us']:9.1f} us" if res["ok"] else ""
        print(f"{status}  {res['suite']:<9} {res['id']:<34} {timing}  {res['detail']}", flush=True)

    print("\ncategory      " + "  ".join(f"{s:>10}" for s in SUITE_NAMES))
    for letter, name in CATEGORIES.items():
        cells = []
        for suite in SUITE_NAMES:
            rs = [r for r in results if r["suite"] == suite and r["id"].startswith(letter)]
            cells.append(f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-")
        print(f"{name:<13} " + "  ".join(f"{c:>10}" for c in cells))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    if args.markdown:
        args.markdown.write_text(markdown_table(results))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
