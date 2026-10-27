"""Problem generation and execution for benchmarks, using the organiser contract."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tools.generator_support import emmy_executable, validate_package
from tools.launch_contract import ordered_kernels

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i32": np.int32, "i64": np.int64}


def generator(checkout: Path):
    path = checkout / "tools" / "generate.py"
    spec = importlib.util.spec_from_file_location("organiser_generator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(spec: dict, destination: Path, checkout: Path):
    mod = generator(checkout)
    emmy = emmy_executable()
    with tempfile.TemporaryDirectory() as td:
        dump = Path(td) / "dump"
        mod.trace(str(emmy), spec, dump, "sm_89")
        row = mod.package(spec, dump, destination)
        validate_package(destination)
        return row


def compile_submission(ir: Path):
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True, check=True)
    return proc.stdout, (time.perf_counter() - start) * 1000


def input_array(arg: dict):
    dtype = DTYPES[arg["dtype"]]
    shape = tuple(arg["shape"])
    if "data" in arg:
        data = np.asarray(arg["data"], dtype=dtype)
        if data.size == 1:
            return np.full(shape, data.item(), dtype=dtype)
        return data.reshape(shape)
    return np.random.default_rng(arg["seed"]).standard_normal(shape).astype(dtype)


def launch_args(kernel: dict, produced: dict, inputs: dict | None = None):
    """Copy descriptors without mutating meta; wire every intermediate explicitly."""
    args = []
    descriptors = {a["name"]: a for a in kernel["args"]}
    for name in kernel["arg_order"]:
        arg = dict(descriptors[name])
        intermediate = arg.pop("intermediate", False)
        if intermediate:
            arg["data"] = np.asarray(produced[name]).ravel().tolist()
            arg.pop("seed", None)
        elif inputs is not None and name in inputs and not arg.get("output"):
            arg["data"] = np.asarray(inputs[name]).ravel().tolist()
            arg.pop("seed", None)
        args.append(arg)
    return args


def check_outputs(produced: dict, expected: dict, meta: dict):
    if not expected:
        raise ValueError("No expected outputs; correctness cannot be established")
    errors = {}
    for name, reference in expected.items():
        if name not in produced:
            raise ValueError(f"Missing final output {name}")
        got = np.asarray(produced[name]).reshape(reference.shape)
        delta = np.abs(got.astype(np.float64) - reference.astype(np.float64))
        errors[name] = dict(max_abs=float(delta.max()), max_rel=float((delta / np.maximum(np.abs(reference), 1e-8)).max()))
        if not np.isfinite(got).all() or not np.allclose(got, reference, rtol=meta["rtol"], atol=meta["atol"]):
            raise ValueError(f"Numerical mismatch {name}: {errors[name]}, rtol={meta['rtol']}, atol={meta['atol']}")
    return errors


def execute_remote(meta, ptxs, runner, *, timing=True, inputs=None, return_final=True):
    produced, times = {}, []
    for index, (kernel, ptx) in enumerate(zip(meta["kernels"], ptxs, strict=True)):
        compiled = runner.compile(ptx)
        if not compiled.get("ok"):
            raise RuntimeError(f"assembly: {compiled}")
        if kernel["entry"] not in compiled["entry_points"]:
            raise ValueError("Runner did not return the expected entry point")
        need = sum(math.prod(a["shape"]) for a in kernel["args"] if a.get("output"))
        if need > 2**22:
            raise ValueError("Runner max_return limit is 2^22 values; full correctness check unavailable")
        later_inputs = {a["name"] for k in meta["kernels"][index+1:] for a in k["args"] if a.get("intermediate")}
        must_return = return_final or any(a["name"] in later_inputs for a in kernel["args"] if a.get("output"))
        result = runner.run(dict(cubin_id=compiled["cubin_id"], entry=kernel["entry"], grid=kernel["grid"],
                                 block=kernel["block"], smem_bytes=kernel["smem_bytes"], zero_outputs=kernel["zero_outputs"],
                                 args=launch_args(kernel, produced, inputs), max_return=max(1, need) if must_return else 1,
                                 warmup=3 if timing else 0, iters=20 if timing else 1))
        if not result.get("ok"):
            raise RuntimeError(f"{kernel['entry']}: execution: {result}")
        times.append(float(result["median_us"]))
        for name, output in result["outputs"].items():
            if not must_return:
                continue
            if output.get("truncated") or len(output["values"]) != math.prod(output["shape"]):
                raise ValueError(f"Truncated output: {name}")
            produced[name] = np.asarray(output["values"], dtype=DTYPES[output["dtype"]]).reshape(output["shape"])
    return produced, times


def read_problem(path):
    meta = json.loads((path / "meta.json").read_text())
    meta["kernels"] = ordered_kernels(meta)
    with np.load(path / "expected.npz") as archive:
        expected = dict(archive)
    return meta, expected
