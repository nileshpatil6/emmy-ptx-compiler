#!/usr/bin/env python3
"""Compare EmmyPTX with Emmy + NVRTC, not an nvcc executable.

Run: python bench/compare_nvcc.py [--only vector_add] [--compile-repeats 7]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.common import ROOT, check_outputs, compile_submission, execute_remote, generate, read_problem  # noqa: E402
from bench.nvrtc_backend import NVRTC, OPTIONS, render_cuda  # noqa: E402
from bench.workloads import workloads  # noqa: E402
from emmyptx import compile_ir  # noqa: E402
from emmyptx.loader import load_kernel  # noqa: E402
from emmyptx.lower import lower_kernel  # noqa: E402
from tools.runner import Runner  # noqa: E402


def measured(fn, repeats):
    fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return dict(median=statistics.median(values), samples=values)


def compile_metrics(ir, nvrtc, repeats):
    source = render_cuda(ir)
    spec = load_kernel(ir)
    nvrtc.compile(source)
    own_ptx, _ = compile_submission(ir)
    cuda_ptx, _ = nvrtc.compile(source)
    metrics = {
        "emmyptx_process_ms": measured(lambda: compile_submission(ir), repeats),
        "nvrtc_process_ms": measured(lambda: subprocess.run(
            [sys.executable, "-m", "bench.nvrtc_backend", str(ir)], cwd=ROOT,
            capture_output=True, check=True), repeats),
        "emmyptx_ir_ms": measured(lambda: compile_ir(ir), repeats),
        "emmyptx_lower_ms": measured(lambda: lower_kernel(spec), repeats),
    }
    samples = [nvrtc.compile(source)[1] for _ in range(repeats)]
    metrics["nvrtc_call_ms"] = dict(median=statistics.median(samples), samples=samples)
    metrics["ir_sha256"] = hashlib.sha256(ir.read_bytes()).hexdigest()
    metrics["cuda_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return own_ptx, cuda_ptx, metrics


def report(payload, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    plt.style.use("default")
    rows = payload["results"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    families = ("vector_add", "exp", "row_sum", "softmax", "matmul", "attention")
    labels = ("Elements", "Elements", "Row width (128 rows)", "Row width (128 rows)", "Square matrix side", "Sequence length (head 64)")
    for ax, family, xlabel in zip(axes.flat, families, labels, strict=True):
        rs = sorted([r for r in rows if r["family"] == family and r["status"] == "pass"], key=lambda r: r["size"])
        for backend, label in (("emmyptx", "EmmyPTX"), ("nvrtc", "Emmy + NVRTC")):
            ax.plot([r["size"] for r in rs], [r["execution_us"][backend] for r in rs], "o-", label=label)
        ax.set(title=family.replace("_", " "), xlabel=xlabel, ylabel="Sum of kernel medians (us)", xscale="log", yscale="log")
        if rs:
            ticks = [r["size"] for r in rs]
            tick_labels = ([f"$2^{{{size.bit_length()-1}}}$" for size in ticks]
                           if family in ("vector_add", "exp") else [str(size) for size in ticks])
            ax.set_xticks(ticks, tick_labels, fontsize=9)
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        if not rs:
            ax.set_xlim(1, 10)
            ax.set_ylim(1, 10)
            ax.text(0.5, 0.5, "No validated measurements", transform=ax.transAxes, ha="center")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"size_sweep.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)
    valid = [r for r in rows if r["status"] == "pass"]
    summary_keys = ("emmyptx_process_ms", "nvrtc_process_ms", "emmyptx_ir_ms", "emmyptx_lower_ms", "nvrtc_call_ms")
    summary = {key: statistics.median(sum(k[key]["median"] for k in r["compile"]) for r in valid)
               for key in summary_keys} if valid else {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    representatives = [max((r for r in valid if r["family"] == family), key=lambda r: r["size"])
                       for family in families if any(r["family"] == family for r in valid)]
    names = [f"{r['family']}\n{r['size']}" for r in representatives]
    positions = list(range(len(representatives)))
    fig.suptitle("Compile time at the largest validated size in each family")
    for ax, metrics, title in zip(axes, (("emmyptx_process_ms", "nvrtc_process_ms"), ("emmyptx_lower_ms", "nvrtc_call_ms")),
                                ("Fresh process: IR to PTX", "Preloaded lowering / NVRTC compile call"), strict=True):
        for offset, key, label in zip((-0.2, 0.2), metrics, ("EmmyPTX", "Emmy + NVRTC"), strict=True):
            values = [sum(k[key]["median"] for k in r["compile"]) for r in representatives]
            ax.bar([p + offset for p in positions], values, width=0.4, label=label)
        ax.set(title=title, ylabel="Sum of per-kernel compile medians (ms)",
               yscale="log" if metrics[0] == "emmyptx_lower_ms" else "linear")
        ax.set_xticks(positions, names, rotation=45, ha="right", fontsize=9)
        ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"compile_times.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)
    lines = ["# EmmyPTX versus Emmy + NVRTC", "", "Actual nvcc was not measured. Both paths compile the same scheduled Kernel IR.", "",
             f"Started {payload['timestamp']}; runner GPU: {payload['runner']['gpu']}. NVRTC options: `{list(OPTIONS)}`.",
             "No fast math; implicit CUDA multiply-add contraction disabled. EmmyPTX still fuses explicit reduction products.", "",
             "CUDA parameter declarations are reordered to match meta.json; the rendered kernel body and schedule are unchanged.", "",
             "Fresh-process timings include startup, imports, IR loading, rendering and PTX retrieval. Warm IR timings include parsing;",
             "pure lowering uses a preloaded specification. NVRTC call timing excludes rendering, program creation and PTX retrieval.",
             f"Compile: one warm-up, {payload['compile_repeats']} samples. GPU: {payload['rounds']} alternating paired rounds,",
             "each with 3 warm-ups and 20 launches. Client response caching is disabled. Clocks are not locked.",
             "Both outputs are fully checked. Timing rounds may omit final output transfers after validation; intermediates are retained.",
             "Execution values sum kernel medians; they exclude transfers and are not end-to-end model latency.", "",
             "![Execution size sweep](size_sweep.png)", "", "![Compile times](compile_times.png)", "",
             "| Workload | Size | EmmyPTX us | NVRTC us | NVRTC / EmmyPTX |",
             "|---|---:|---:|---:|---:|"]
    for r in valid:
        a, b = r["execution_us"]["emmyptx"], r["execution_us"]["nvrtc"]
        lines.append(f"| {r['family']} | {r['size']} | {a:.3f} | {b:.3f} | {b/a:.3f} |")
    lines += ["", "A ratio above 1 favours EmmyPTX. Differences near 1 can be measurement noise.", "",
              "| Workload | Process EmmyPTX / NVRTC ms | Warm IR ms | Lowering / NVRTC call ms |",
              "|---|---:|---:|---:|"]
    for r in valid:
        def total(key, row=r):
            return sum(k[key]["median"] for k in row["compile"])
        lines.append(f"| {r['id']} | {total('emmyptx_process_ms'):.3f} / {total('nvrtc_process_ms'):.3f} | "
                     f"{total('emmyptx_ir_ms'):.3f} | {total('emmyptx_lower_ms'):.3f} / {total('nvrtc_call_ms'):.3f} |")
    if summary:
        lines += ["", "## Compile summary", "",
                  f"Median across the {len(valid)} validated workload medians (each sums its kernels), in ms:", "",
                  "| Measurement | EmmyPTX | Emmy + NVRTC |", "|---|---:|---:|",
                  f"| Fresh process, IR to PTX | {summary['emmyptx_process_ms']:.3f} | {summary['nvrtc_process_ms']:.3f} |",
                  f"| Warm IR compilation | {summary['emmyptx_ir_ms']:.3f} | not measured separately |",
                  f"| Preloaded lowering / NVRTC compile call | {summary['emmyptx_lower_ms']:.3f} | {summary['nvrtc_call_ms']:.3f} |",
                  "", "The last row measures different compilation stages; it does not include startup or rendering."]
    lines += ["", "## Skips and failures", ""]
    for r in rows:
        if r["status"] != "pass":
            lines.append(f"- {r['id']}: {r['status']}: {r.get('reason', '').replace(chr(10), ' ')}")
    if valid:
        ratios = [r["execution_us"]["nvrtc"] / r["execution_us"]["emmyptx"] for r in valid]
        lines += ["", f"Across validated sizes, NVRTC / EmmyPTX execution ratios range from {min(ratios):.3f} to {max(ratios):.3f}.",
                  "These measurements compare code generation for fixed schedules, not independently tuned libraries.",
                  "No causal explanation for a speed difference is claimed without profiling."]
    lines += ["", "Exact samples, correctness errors, versions, hashes and failures: [raw_results.json](raw_results.json).", ""]
    (out / "RESULTS_nvcc.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hackathon", type=Path, default=ROOT.parent / "emmy-hackathon-2026")
    ap.add_argument("--only", default="")
    ap.add_argument("--compile-repeats", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "bench")
    ap.add_argument("--reuse-generated", action="store_true", help="reuse cached workload fixtures explicitly")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume completed measurements with unchanged options")
    ap.add_argument("--retry-failed", action="store_true", help="retry failed cases in a resumed run, retaining their diagnostics")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(json.loads((args.out / "raw_results.json").read_text()), args.out)
        return 0
    if args.compile_repeats < 1 or args.rounds < 1:
        ap.error("Repeat counts must be positive")
    nvrtc, runner = NVRTC(), Runner(use_cache=False)
    conn = runner._connection()
    try:
        conn.request("GET", "/health")
        health = json.loads(conn.getresponse().read())
    finally:
        conn.close()
    payload = dict(timestamp=datetime.now(UTC).isoformat(), runner=health, nvrtc=nvrtc.version,
                   compile_repeats=args.compile_repeats, rounds=args.rounds, options=list(OPTIONS),
                   versions={n: importlib.metadata.version(n) for n in ("emmy-ml", "numpy", "torch", "nvidia-cuda-nvrtc-cu12")},
                   generator_commit=subprocess.check_output(["git", "-C", str(args.hackathon), "rev-parse", "HEAD"], text=True).strip(),
                   results=[])
    checkpoint = args.out / "raw_results.json"
    if args.resume and checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        for key in ("compile_repeats", "rounds", "versions", "generator_commit", "nvrtc", "options"):
            if previous[key] != payload[key]:
                ap.error(f"Cannot resume: {key} changed")
        payload = previous
    if args.retry_failed:
        payload.setdefault("previous_failures", []).extend(r for r in payload["results"] if r["status"] == "failed")
        payload["results"] = [r for r in payload["results"] if r["status"] != "failed"]
    completed = {r["id"] for r in payload["results"]}
    for spec in workloads():
        if args.only and args.only not in spec["id"]:
            continue
        if spec["id"] in completed:
            continue
        row = {k: spec[k] for k in ("id", "family", "size")}
        row["status"] = "failed"
        path = ROOT / ".cache" / "bench" / spec["id"]
        try:
            print(f"START   {spec['id']}", flush=True)
            if spec["family"] in ("vector_add", "exp") and spec["size"] > 2**22:
                row.update(status="skipped", reason="Full output exceeds runner max_return=2^22; no correctness-validated speed claim")
            else:
                row["stage"] = "generation"
                if not args.reuse_generated or not (path / "meta.json").exists():
                    generate(spec, path, args.hackathon)
                meta, expected = read_problem(path)
                ptxs, row["compile"] = {"emmyptx": [], "nvrtc": []}, []
                row["stage"] = "compilation"
                for kernel in meta["kernels"]:
                    own, cuda, metrics = compile_metrics(path / f"{kernel['entry']}.ir.json", nvrtc, args.compile_repeats)
                    ptxs["emmyptx"].append(own)
                    ptxs["nvrtc"].append(cuda)
                    row["compile"].append(dict(entry=kernel["entry"], **metrics))
                row["stage"] = "execution"
                samples = {"emmyptx": [], "nvrtc": []}
                row["errors"] = {}
                for backend in ("emmyptx", "nvrtc"):
                    print(f"  validate {backend}", flush=True)
                    produced, _ = execute_remote(meta, ptxs[backend], runner, timing=False)
                    row["errors"][backend] = check_outputs(produced, expected, meta)
                for round_index in range(args.rounds):
                    order = ("emmyptx", "nvrtc") if round_index % 2 == 0 else ("nvrtc", "emmyptx")
                    for backend in order:
                        _, times = execute_remote(meta, ptxs[backend], runner, return_final=False)
                        samples[backend].append(times)
                row.update(status="pass", samples_us=samples,
                           execution_us={k: statistics.median(sum(t) for t in v) for k, v in samples.items()})
        except Exception as exc:
            row["reason"] = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            if isinstance(row["reason"], bytes):
                row["reason"] = row["reason"].decode(errors="replace")
        payload["results"].append(row)
        (args.out / "raw_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{row['status'].upper():7} {spec['id']} {row.get('execution_us', row.get('reason', ''))}", flush=True)
    report(payload, args.out)
    return int(any(r["status"] == "failed" for r in payload["results"]))


if __name__ == "__main__":
    sys.exit(main())
