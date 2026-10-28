#!/usr/bin/env python3
"""Run a pre-norm transformer block as small, separately traced Emmy stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
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
    read_problem,
)
from bench.demo_e2e import local_execute  # noqa: E402
from tools.runner import Runner  # noqa: E402

SEQ, D_MODEL, D_FF = 8, 32, 64
RTOL, ATOL = 1e-3, 1e-4
SEED = 2026
HEADER = "EmmyPTX demo | pre-norm transformer block | seq 8, d_model 32, d_ff 64 | causal attention"

WHOLE_BLOCK = """class WholeBlock(nn.Module):
    def forward(self, x, ln1_weight, ln1_bias, wq, wk, wv, wo,
                ln2_weight, ln2_bias, w1, w3, w2, mask):
        n = F.layer_norm(x, (32,), ln1_weight, ln1_bias, eps=1e-5)
        q, k, v = n @ wq, n @ wk, n @ wv
        attention = torch.softmax(q @ k.transpose(-1, -2) / 32**0.5 + mask, -1) @ v
        h = x + attention @ wo
        n = F.layer_norm(h, (32,), ln2_weight, ln2_bias, eps=1e-5)
        return h + (F.silu(n @ w1) * (n @ w3)) @ w2
"""
WHOLE_INPUTS = ("x", "ln1_weight", "ln1_bias", "wq", "wk", "wv", "wo",
                "ln2_weight", "ln2_bias", "w1", "w3", "w2", "mask")


@dataclass(frozen=True)
class Stage:
    name: str
    # Each pair maps a module's positional parameter to the block's named buffer.
    inputs: tuple[tuple[str, str], ...]
    shapes: tuple[tuple[int, ...], ...]
    expression: str
    output: str
    knobs: str = ""

    @property
    def source(self):
        signature = ", ".join(name for name, _ in self.inputs)
        return f"class Stage(nn.Module):\n def forward(self, {signature}):\n  return {self.expression}"


def stages():
    matrix, vector, weight = (SEQ, D_MODEL), (D_MODEL,), (D_MODEL, D_MODEL)
    norm_knobs = "WORK=t128,REDUCE=coop"
    return [
        Stage("ln1", (("x", "x"), ("weight", "ln1_weight"), ("bias", "ln1_bias")),
              (matrix, vector, vector), "F.layer_norm(x, (32,), weight, bias, eps=1e-5)", "norm1", norm_knobs),
        *(Stage(f"{letter}_projection", (("x", "norm1"), ("weight", f"w{letter}")),
                (matrix, weight), "x @ weight", letter) for letter in ("q", "k", "v")),
        Stage("attention_core", (("q", "q"), ("kt", "kt"), ("v", "v"), ("mask", "mask")),
              (matrix, (D_MODEL, SEQ), matrix, (SEQ, SEQ)),
              "torch.softmax(q @ kt / 32**0.5 + mask, -1) @ v", "attention"),
        Stage("attention_residual", (("x", "attention"), ("weight", "wo"), ("residual", "x")),
              (matrix, weight, matrix), "residual + x @ weight", "h"),
        Stage("ln2", (("x", "h"), ("weight", "ln2_weight"), ("bias", "ln2_bias")),
              (matrix, vector, vector), "F.layer_norm(x, (32,), weight, bias, eps=1e-5)", "norm2", norm_knobs),
        Stage("swiglu_residual", (("x", "norm2"), ("gate", "w1"), ("up", "w3"), ("down", "w2"), ("residual", "h")),
              (matrix, (D_MODEL, D_FF), (D_MODEL, D_FF), (D_FF, D_MODEL), matrix),
              "residual + (F.silu(x @ gate) * (x @ up)) @ down", "out"),
    ]


def map_stage_inputs(meta, signature, arrays):
    """Bind a verified module signature to its exact float32 launch inputs."""
    names = list(meta["graph_inputs"])
    if names != list(signature):
        raise ValueError(f"Graph input order {names} differs from module signature {list(signature)}")
    if len(arrays) != len(names):
        raise ValueError(f"Expected {len(names)} input arrays, got {len(arrays)}")
    descriptors = {}
    for kernel in meta["kernels"]:
        for arg in kernel["args"]:
            if arg["name"] in names:
                shape_dtype = (tuple(arg["shape"]), arg["dtype"])
                if arg["name"] in descriptors and descriptors[arg["name"]] != shape_dtype:
                    raise ValueError(f"Inconsistent launch descriptors for {arg['name']}")
                descriptors[arg["name"]] = shape_dtype
    bound = {}
    for name, array in zip(names, arrays, strict=True):
        if name not in descriptors:
            raise ValueError(f"Graph input {name} has no launch descriptor")
        shape, dtype = descriptors[name]
        if not isinstance(array, np.ndarray) or array.shape != shape:
            raise ValueError(f"Input {name} must have shape {shape}; got {getattr(array, 'shape', None)}")
        if array.dtype != np.dtype(DTYPES[dtype]):
            raise ValueError(f"Input {name} must have dtype {dtype}; got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"Input {name} contains nonfinite values")
        bound[name] = np.ascontiguousarray(array)
    return bound


def initial_buffers():
    rng = np.random.default_rng(SEED)

    def random(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    buffers = {"x": random((SEQ, D_MODEL), 0.2)}
    for prefix in ("ln1", "ln2"):
        buffers[f"{prefix}_weight"] = np.ones(D_MODEL, np.float32) + random((D_MODEL,), 0.05)
        buffers[f"{prefix}_bias"] = random((D_MODEL,), 0.01)
    for name in ("wq", "wk", "wv", "wo"):
        buffers[name] = random((D_MODEL, D_MODEL), 0.1)
    buffers["w1"] = random((D_MODEL, D_FF), 0.1)
    buffers["w3"] = random((D_MODEL, D_FF), 0.1)
    buffers["w2"] = random((D_FF, D_MODEL), 0.1)
    # A host tensor is passed to the attention stage, never folded by Emmy.
    buffers["mask"] = np.where(np.arange(SEQ)[:, None] >= np.arange(SEQ)[None, :], 0.0, -1e9).astype(np.float32)
    return buffers


class CompilerStageError(RuntimeError):
    """Compilation failure: stop without editing the compiler or falling back."""


class StageNumericalError(RuntimeError):
    """A numerical mismatch must never trigger a backend fallback."""


def torch_module(source, class_name="Stage"):
    import torch

    scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(source, scope)
    return scope[class_name]().eval()


def reference_call(module, arrays):
    import torch

    with torch.no_grad():
        return module(*(torch.from_numpy(np.ascontiguousarray(a)) for a in arrays)).numpy()


def prepare(checkout, reuse):
    prepared = []
    for stage in stages():
        directory = ROOT / ".cache" / "demo_transformer" / stage.name
        module = torch_module(stage.source)
        constructors = ", ".join(f"torch.randn({', '.join(str(d) for d in shape)})" for shape in stage.shapes)
        code = f"exec({stage.source!r}); Stage()({constructors})"
        spec = dict(id=stage.name, code=code, ref=lambda *xs, model=module: [reference_call(model, xs)],
                    rtol=RTOL, atol=ATOL, knobs=stage.knobs)
        if not reuse or not (directory / "meta.json").exists():
            generate(spec, directory, checkout)
        meta, _ = read_problem(directory)
        if meta["code"] != code:
            raise ValueError(f"{stage.name}: cached trace differs; rerun without --reuse-generated")
        if len(meta["graph_outputs"]) != 1:
            raise ValueError(f"{stage.name}: expected exactly one stage output")
        ptxs, compile_ms = [], []
        for kernel in meta["kernels"]:
            ir = directory / f"{kernel['entry']}.ir.json"
            try:
                ptx, ms = compile_submission(ir)
            except subprocess.CalledProcessError as exc:
                raise CompilerStageError(f"{stage.name}: compiler failed for {ir}\n{exc.stderr}") from exc
            ptxs.append(ptx)
            compile_ms.append(ms)
        prepared.append((stage, directory, module, meta, ptxs, compile_ms))
    return prepared


def execute(prepared, original, backend, emit):
    state = {name: value.copy() for name, value in original.items()}
    runner = Runner(use_cache=False) if backend == "remote" else None
    rows, device = [], None
    emit(f"Backend: {backend}")
    for stage, directory, module, meta, ptxs, compile_ms in prepared:
        if stage.name == "attention_core":
            # Layout-only host hand-off of the compiled K projection.
            state["kt"] = np.ascontiguousarray(state["k"].T)
        arrays = [state[name] for _, name in stage.inputs]
        inputs = map_stage_inputs(meta, [name for name, _ in stage.inputs], arrays)
        expected = {meta["graph_outputs"][0]: reference_call(module, arrays)}
        try:
            if backend == "local":
                produced, times, device = local_execute(meta, ptxs, inputs, directory)
            else:
                produced, times = execute_remote(meta, ptxs, runner, inputs=inputs)
                device = "organiser runner (RTX 5090)"
        except Exception as exc:
            raise RuntimeError(f"{stage.name}: execution failed; IR folder {directory}: {exc}") from exc
        try:
            errors = check_outputs(produced, expected, meta)
        except ValueError as exc:
            raise StageNumericalError(f"{stage.name}: {exc}; IR folder {directory}") from exc
        output = np.asarray(produced[meta["graph_outputs"][0]], dtype=np.float32)
        state[stage.output] = output
        rows.append(dict(stage=stage.name, kernels=len(ptxs), compile_ms=sum(compile_ms), gpu_us=sum(times),
                         kernel_compile_ms=compile_ms, kernel_gpu_us=times, errors=errors))
        emit(f"  {stage.name}: {len(ptxs)} kernel(s) | compile {sum(compile_ms):.2f} ms | GPU {sum(times):.3f} us")
    return state["out"], rows, device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("auto", "local", "remote"), default="auto")
    ap.add_argument("--reuse-generated", action="store_true")
    ap.add_argument("--hackathon", type=Path, default=ROOT.parent / "emmy-hackathon-2026")
    args = ap.parse_args()
    lines = []

    def emit(line):
        lines.append(line)
        print(line, flush=True)

    emit(HEADER)
    original = initial_buffers()
    # This reference runs the entire block from the original arrays, independently
    # of all compiled stage outputs and host hand-offs.
    expected = reference_call(torch_module(WHOLE_BLOCK, "WholeBlock"), [original[n] for n in WHOLE_INPUTS])
    prepared = prepare(args.hackathon, args.reuse_generated)
    backend = "remote" if args.backend == "remote" else "local"
    fallback = None
    try:
        output, rows, device = execute(prepared, original, backend, emit)
    except StageNumericalError:
        raise
    except RuntimeError as exc:
        if args.backend != "auto":
            raise
        fallback = str(exc)
        emit(f"Local execution unavailable: {fallback}")
        backend = "remote"
        output, rows, device = execute(prepared, original, backend, emit)
    try:
        errors = check_outputs({"out": output}, {"out": expected}, dict(rtol=RTOL, atol=ATOL))["out"]
    except ValueError as exc:
        raise StageNumericalError(f"Whole block: {exc}") from exc
    emit(f"Device: {device}")
    emit(f"Max absolute error vs PyTorch full block: {errors['max_abs']:.6g}")
    emit(f"Max relative error vs PyTorch full block (floor 1e-8): {errors['max_rel']:.6g}")
    emit("PASS: transformer block output matches PyTorch")
    report_path = ROOT / "docs" / "demo_transformer_result.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(config=dict(seq=SEQ, d_model=D_MODEL, d_ff=D_FF, heads=1, seed=SEED,
                              rtol=RTOL, atol=ATOL, mask="additive causal input: 0 / -1e9"), last_backend=backend,
                  timing_note="Stage GPU times sum kernel medians; host hand-offs, transfers and startup are excluded.")
    report.setdefault("runs", {})[backend] = dict(status="pass", device=device, fallback=fallback, stages=rows,
                                                 errors=errors, output=output.tolist(), console_output="\n".join(lines) + "\n")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
