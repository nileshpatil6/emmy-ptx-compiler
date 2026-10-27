#!/usr/bin/env python3
"""Trace a PyTorch MLP, compile its kernels with EmmyPTX, and run on the GPU."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.common import (  # noqa: E402
    DTYPES,
    ROOT,
    check_outputs,
    compile_submission,
    execute_remote,
    generate,
    input_array,
    read_problem,
)
from tools.runner import Runner  # noqa: E402

MODEL_SOURCE = """class TinyMLP(nn.Module):
    def forward(self, x, w1, b1, w2, b2):
        hidden = torch.relu(x @ w1 + b1)
        return hidden, hidden @ w2 + b2
"""


def model_spec():
    # Exposing hidden as a second observable also checks intermediates against PyTorch.
    code = f"exec({MODEL_SOURCE!r}); TinyMLP()(torch.randn(8, 32), torch.randn(32, 64), torch.randn(64), "
    code += "torch.randn(64, 10), torch.randn(10))"

    def reference(x, w1, b1, w2, b2):
        hidden = np.maximum(x.astype(np.float64) @ w1 + b1, 0)
        return [hidden.astype(np.float32), (hidden @ w2 + b2).astype(np.float32)]

    return dict(id="m_demo_mlp", code=code, ref=reference, rtol=1e-3, atol=1e-4, knobs="PLACE=cut")


def laptop_ptx(ptx):
    result, count = re.subn(r"(?m)^\.target sm_89\s*$", ".target sm_86", ptx)
    if count != 1:
        raise ValueError("Expected exactly one sm_89 target directive")
    return result


def local_execute(meta, ptxs, inputs, directory):
    import warnings

    # PTX loading uses the driver, so absence of a toolkit is expected.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA path could not be detected.*")
        import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(0)
    if (device["major"], device["minor"]) != (8, 6):
        raise RuntimeError("This demo's local target override is only for compute capability 8.6")
    stream = cp.cuda.Stream.null
    buffers = {name: cp.asarray(value) for name, value in inputs.items()}
    modules, times = [], []
    for index, (kernel, ptx) in enumerate(zip(meta["kernels"], ptxs, strict=True)):
        path = directory / f"local_{index}.ptx"
        path.write_text(laptop_ptx(ptx), encoding="ascii")
        module = cp.RawModule(path=str(path))
        modules.append(module)
        function = module.get_function(kernel["entry"])
        zeros = {}
        for arg in kernel["args"]:
            name = arg["name"]
            if arg.get("output"):
                # Host zeros + transfer avoids invoking a CuPy-generated fill kernel.
                zeros[name] = np.zeros(arg["shape"], dtype=DTYPES[arg["dtype"]])
                buffers[name] = cp.asarray(zeros[name])
            elif name not in buffers:
                if arg.get("intermediate"):
                    raise ValueError(f"Missing intermediate {name}")
                buffers[name] = cp.asarray(input_array(arg))
        call_args = tuple(buffers[name] for name in kernel["arg_order"])
        durations = []
        for repetition in range(23):
            for name in kernel["zero_outputs"]:
                buffers[name].set(zeros[name], stream=stream)
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            function(tuple(kernel["grid"]), tuple(kernel["block"]), call_args,
                     shared_mem=kernel["smem_bytes"], stream=stream)
            end.record(stream)
            end.synchronize()
            if repetition >= 3:
                durations.append(cp.cuda.get_elapsed_time(start, end) * 1000)
        times.append(statistics.median(durations))
    produced = {name: cp.asnumpy(buffers[name]) for name in meta["graph_outputs"]}
    return produced, times, device["name"].decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hackathon", type=Path, default=ROOT.parent / "emmy-hackathon-2026")
    ap.add_argument("--backend", choices=("auto", "local", "remote"), default="auto")
    ap.add_argument("--reuse-generated", action="store_true")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "demo_result.json")
    args = ap.parse_args()
    directory = ROOT / ".cache" / "demo"
    print("EmmyPTX demo | PyTorch MLP 32 -> 64 -> 10 | batch 8", flush=True)
    if not args.reuse_generated or not (directory / "meta.json").exists():
        generate(model_spec(), directory, args.hackathon)
    meta, _ = read_problem(directory)
    import torch

    scope = {"torch": torch, "nn": torch.nn}
    exec(MODEL_SOURCE, scope)
    inputs = {}
    for kernel in meta["kernels"]:
        for arg in kernel["args"]:
            if arg["name"] in meta["graph_inputs"]:
                inputs[arg["name"]] = input_array(arg)
    with torch.no_grad():
        outputs = scope["TinyMLP"]().eval()(*(torch.from_numpy(inputs[name]) for name in meta["graph_inputs"]))
    expected = {name: tensor.numpy() for name, tensor in zip(meta["graph_outputs"], outputs, strict=True)}
    ptxs, compile_ms = [], []
    for kernel in meta["kernels"]:
        ptx, ms = compile_submission(directory / f"{kernel['entry']}.ir.json")
        ptxs.append(ptx)
        compile_ms.append(ms)
    fallback = None
    if args.backend != "remote":
        try:
            produced, times, device = local_execute(meta, ptxs, inputs, directory)
            backend = "local"
        except Exception as exc:
            if args.backend == "local":
                raise
            fallback = f"{type(exc).__name__}: {exc}"
            print(f"Local execution unavailable: {fallback}\nUsing remote runner.", flush=True)
    if args.backend == "remote" or fallback is not None:
        produced, times = execute_remote(meta, ptxs, Runner(use_cache=False), inputs=inputs)
        backend, device = "remote", "organiser runner (RTX 5090)"
    # Correctness failures are never treated as local device incompatibilities.
    errors = check_outputs(produced, expected, meta)
    print(f"Backend: {backend} | {device}")
    for kernel, ms, us in zip(meta["kernels"], compile_ms, times, strict=True):
        print(f"  {kernel['entry']}: compile {ms:.2f} ms | GPU {us:.3f} us")
    logits = produced[meta["graph_outputs"][-1]]
    print("Logits:\n" + np.array2string(logits, precision=4, suppress_small=True))
    print("Predicted classes: " + str(np.argmax(logits, axis=-1).tolist()))
    print(f"Max absolute error: {max(e['max_abs'] for e in errors.values()):.6g}")
    print(f"Max relative error (denominator floor 1e-8): {max(e['max_rel'] for e in errors.values()):.6g}")
    result = dict(backend=backend, device=device, fallback=fallback, errors=errors, compile_ms=compile_ms,
                  kernel_us=times, kernels=[k["entry"] for k in meta["kernels"]],
                  rtol=meta["rtol"], atol=meta["atol"], logits=logits.tolist(), status="pass")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS: all model outputs match PyTorch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
