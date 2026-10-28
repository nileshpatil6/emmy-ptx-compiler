"""Offline checks for correctness-critical benchmark and demo behavior."""

import ast
from pathlib import Path

import numpy as np
import pytest

from bench.common import check_outputs, execute_remote, input_array, launch_args
from bench.compare_nvcc import measured
from bench.demo_e2e import laptop_ptx
from bench.demo_transformer import map_stage_inputs
from bench.nvrtc_backend import render_cuda
from tools.launch_contract import ordered_kernels


def test_inputs_match_organiser_dtype_and_seed():
    arg = dict(name="x", dtype="i64", shape=[7], seed=2)
    assert np.array_equal(input_array(arg), np.random.default_rng(2).standard_normal(7).astype(np.int64))
    assert np.array_equal(input_array(dict(dtype="f32", shape=[3], data=[2])), [2, 2, 2])


def test_stage_input_mapping_checks_names_order_shape_and_dtype():
    meta = dict(graph_inputs=["q", "k"], kernels=[dict(args=[
        dict(name="k", shape=[2, 2], dtype="f32"), dict(name="q", shape=[2, 2], dtype="f32")])])
    q = np.arange(4, dtype=np.float32).reshape(2, 2)
    k = np.ones((2, 2), dtype=np.float32).T
    bound = map_stage_inputs(meta, ("q", "k"), [q, k])
    assert np.array_equal(bound["q"], q)
    assert np.array_equal(bound["k"], k)
    assert bound["k"].flags.c_contiguous
    with pytest.raises(ValueError, match="input order"):
        map_stage_inputs(meta, ("k", "q"), [k, q])
    with pytest.raises(ValueError, match="shape"):
        map_stage_inputs(meta, ("q", "k"), [q[:, :1], k])
    with pytest.raises(ValueError, match="dtype"):
        map_stage_inputs(meta, ("q", "k"), [q.astype(np.float64), k])
    with pytest.raises(ValueError, match="input arrays"):
        map_stage_inputs(meta, ("q", "k"), [q])


def test_launch_order_and_intermediate_values():
    kernel = dict(arg_order=["middle", "out"], args=[dict(name="out", output=True), dict(name="middle", intermediate=True)])
    args = launch_args(kernel, {"middle": np.array([1, 2], dtype=np.int64)})
    assert args == [dict(name="middle", data=[1, 2]), dict(name="out", output=True)]
    assert kernel["args"][1]["intermediate"]
    with pytest.raises(KeyError):
        launch_args(kernel, {})


def test_remote_chaining_and_atomic_reset_contract():
    def kernel(entry, input_name, output_name, intermediate):
        return dict(entry=entry, grid=[1, 1, 1], block=[32, 1, 1], smem_bytes=0, zero_outputs=[output_name],
                    arg_order=[input_name, output_name], args=[
                        dict(name=input_name, dtype="f32", shape=[2], intermediate=intermediate, seed=0),
                        dict(name=output_name, dtype="f32", shape=[2], output=True)])

    class FakeRunner:
        requests = []

        def compile(self, ptx):
            return dict(ok=True, cubin_id=ptx, entry_points=[ptx])

        def run(self, request):
            self.requests.append(request)
            if request["entry"] == "second":
                assert request["args"][0]["data"] == [4, 5]
                assert "seed" not in request["args"][0]
            name = request["args"][-1]["name"]
            return dict(ok=True, median_us=3, outputs={name: dict(shape=[2], dtype="f32", values=[4, 5], truncated=False)})

    runner = FakeRunner()
    meta = dict(kernels=[kernel("first", "x", "middle", False), kernel("second", "middle", "out", True)])
    produced, times = execute_remote(meta, ["first", "second"], runner)
    assert np.array_equal(produced["out"], [4, 5])
    assert times == [3, 3]
    assert [r["zero_outputs"] for r in runner.requests] == [["middle"], ["out"]]
    assert all(r["warmup"] == 3 and r["iters"] == 20 for r in runner.requests)


def test_correctness_never_passes_missing_or_nonfinite_output():
    meta = dict(rtol=1e-3, atol=1e-4)
    expected = {"x": np.ones(2)}
    for produced in ({}, {"x": np.array([1, np.nan])}, {"x": np.array([1, 2])}):
        with pytest.raises(ValueError):
            check_outputs(produced, expected, meta)
    with pytest.raises(ValueError):
        check_outputs({}, {}, meta)


def test_target_override_is_local_and_exact():
    source = ".version 8.0\n.target sm_89\n.address_size 64\n"
    assert laptop_ptx(source) == source.replace("sm_89", "sm_86")
    assert ".target sm_89" in source
    with pytest.raises(ValueError):
        laptop_ptx(source.replace("sm_89", "sm_90"))


def test_cuda_rendering_honors_metadata_argument_order():
    import json
    import re

    root = Path(__file__).resolve().parents[1] / "tests" / "open" / "m01_matmul_512"
    ir = next(root.glob("*.ir.json"))
    meta = json.loads((root / "meta.json").read_text())
    kernel = meta["kernels"][0]
    source = render_cuda(ir)
    signature = re.search(r"void " + kernel["entry"] + r"\(([^)]*)\)", source).group(1)
    assert [re.search(r"(\w+)\s*$", d).group(1) for d in signature.split(",")] == kernel["arg_order"]


def test_measurement_excludes_warmup_from_samples():
    calls = []
    result = measured(lambda: calls.append(1), 3)
    assert len(calls) == 4
    assert len(result["samples"]) == 3


def test_kernels_are_sorted_by_intermediate_dependencies():
    producer = dict(entry="producer", args=[dict(name="temporary", output=True)])
    consumer = dict(entry="consumer", args=[dict(name="temporary", intermediate=True)])
    assert ordered_kernels(dict(kernels=[consumer, producer])) == [producer, consumer]
    with pytest.raises(ValueError, match="Missing producer"):
        ordered_kernels(dict(kernels=[consumer]))
    cycle = [dict(args=[dict(name="a", output=True), dict(name="b", intermediate=True)]),
             dict(args=[dict(name="b", output=True), dict(name="a", intermediate=True)])]
    with pytest.raises(ValueError, match="Cyclic"):
        ordered_kernels(dict(kernels=cycle))


def test_compact_timing_response_still_returns_intermediates():
    class FakeRunner:
        requests = []

        def compile(self, ptx):
            return dict(ok=True, cubin_id=ptx, entry_points=[ptx])

        def run(self, request):
            self.requests.append(request)
            name = request["args"][-1]["name"]
            return dict(ok=True, median_us=3, outputs={name: dict(shape=[2], dtype="f32", values=[2, 4], truncated=False)})

    first = dict(entry="first", grid=[1]*3, block=[32, 1, 1], smem_bytes=0, zero_outputs=[], arg_order=["mid"],
                 args=[dict(name="mid", dtype="f32", shape=[2], output=True)])
    second = dict(entry="second", grid=[1]*3, block=[32, 1, 1], smem_bytes=0, zero_outputs=[], arg_order=["mid", "out"],
                  args=[dict(name="mid", dtype="f32", shape=[2], intermediate=True),
                        dict(name="out", dtype="f32", shape=[2], output=True)])
    runner = FakeRunner()
    produced, _ = execute_remote(dict(kernels=[first, second]), ["first", "second"], runner, return_final=False)
    assert [r["max_return"] for r in runner.requests] == [2, 1]
    assert runner.requests[1]["args"][0]["data"] == [2, 4]
    assert "out" not in produced


def test_submission_has_no_benchmark_or_external_compiler_imports():
    root = Path(__file__).resolve().parents[1]
    for path in [root / "compile.py", *(root / "emmyptx").glob("*.py")]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(any(part in name.lower() for part in ("bench", "nvrtc", "nvcc", "llvm", "render", "subprocess"))
                           for name in names), (path, names)
