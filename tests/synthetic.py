#!/usr/bin/env python3
"""Hand-built Kernel IR for statement forms emmy rarely emits.

    python tests/synthetic.py        # (re)writes tests/synthetic/<id>/

The generated suites never contain some forms the IR reference lists: a Cond
with an else branch, a segmented TreeHalve, a StridedLoop with an explicit end,
4- and 8-byte cp.async, warp and named barriers, a Tile with producer (aux)
threads, eight-wide f16 vectors, an int64 warp shuffle, Select chains and
integer Lets. These kernels are built from emmy's own statement classes,
serialized the way emmy serializes them (constructor reprs that
``Graph.from_dict`` evaluates back), and packaged like any problem: IR,
meta.json with the launch, and expected.npz computed with numpy from the same
seeded inputs the runner generates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from emmy.compiler import dtype as edt
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Let, Load, Select, SelectBranch, StridedLoop, Write

OUT = Path(__file__).resolve().parent / "synthetic"
F16, F32, I64 = edt.get("f16"), edt.get("f32"), edt.get("i64")
NP = {"f16": np.float16, "f32": np.float32, "i64": np.int64}


def lit(v: int) -> Literal:
    return Literal(v, "int")


def bx(op: str, a, b) -> BinaryExpr:
    return BinaryExpr(op, a, b)


def ax(name: str, n: int) -> Axis:
    return Axis(name, Dim(n))


def seeded(name_shape_dtype: list[tuple[str, tuple, str]]) -> dict[str, np.ndarray]:
    """The runner fills input ``i`` with default_rng(i).standard_normal(shape)."""
    return {n: np.random.default_rng(i).standard_normal(s).astype(NP[d]) for i, (n, s, d) in enumerate(name_shape_dtype)}


def package(pid: str, name: str, body: list, inputs: list, outputs: list, grid: int, block: int, expected: dict, rtol: float, atol: float):
    """Write tests/synthetic/<pid>/ in the hackathon's problem format."""
    tensor = lambda n, s, d: {"name": n, "shape": list(s), "dtype": d}  # noqa: E731
    nodes = {n: {"op": "InputOp", "op_fields": {}, "inputs": [], "outputs": [tensor(n, s, d)]} for n, s, d in inputs}
    out_name = outputs[0][0]
    nodes[out_name] = {
        "op": "KernelOp",
        "op_fields": {"body": [repr(s) for s in body], "name": name, "zero_delegated": [], "serial": []},
        "inputs": [n for n, _, _ in inputs],
        "outputs": [tensor(n, s, d) for n, s, d in outputs],
    }
    graph = {"inputs": [n for n, _, _ in inputs], "outputs": [n for n, _, _ in outputs], "nodes": nodes}
    args = [{"name": n, "dtype": d, "shape": list(s), "seed": i} for i, (n, s, d) in enumerate(inputs)]
    args += [{"name": n, "dtype": d, "shape": list(s), "output": True} for n, s, d in outputs]
    kernel = {
        "node": out_name, "entry": name, "arg_order": [a["name"] for a in args],
        "grid": [grid, 1, 1], "block": [block, 1, 1], "smem_bytes": 0, "zero_outputs": [], "args": args, "final": True,
    }  # fmt: skip
    meta = {"id": pid, "rtol": rtol, "atol": atol, "graph_inputs": graph["inputs"], "graph_outputs": graph["outputs"], "kernels": [kernel]}
    d = OUT / pid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / f"{name}.ir.json").write_text(json.dumps(graph, indent=1))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    np.savez(d / "expected.npz", **expected)


def cond_else_select_let():
    """Cond with else, a three-way Select, square, and an integer Let index."""
    a0 = Var("a0")
    body = [
        Tile(
            axes=(ax("a0", 1000),),
            body=(
                Load(name="v", input="x", index=(a0,), dtype=F32),
                Let(name="j", value=bx("-", lit(999), a0)),
                Load(name="r", input="x", index=(Var("j"),), dtype=F32),
                Assign(name="sq", op="square", args=("v",)),
                Assign(name="ng", op="negative", args=("v",)),
                Select(
                    name="s",
                    branches=(
                        SelectBranch("sq", bx("<", a0, lit(300))),
                        SelectBranch("v", bx("<", a0, lit(600))),
                        SelectBranch("ng", lit(1)),
                    ),
                ),
                Cond(
                    cond=bx("==", bx("%", a0, lit(2)), lit(0)),
                    body=(
                        Assign(name="e", op="exp", args=("s",)),
                        Assign(name="o1", op="add", args=("e", "r")),
                        Write(output="out", index=(a0,), value="o1"),
                    ),
                    else_body=(Assign(name="o2", op="subtract", args=("s", "r")), Write(output="out", index=(a0,), value="o2")),
                ),
            ),
        )
    ]
    x = seeded([("x", (1000,), "f32")])["x"].astype(np.float64)
    i = np.arange(1000)
    s = np.where(i < 300, x * x, np.where(i < 600, x, -x))
    out = np.where(i % 2 == 0, np.exp(s) + x[::-1], s - x[::-1])
    package("p_syn_cond_else_select", "k_syn_cond", body, [("x", (1000,), "f32")], [("out", (1000,), "f32")], 4, 256,
            {"out": out.astype(np.float32)}, 1e-5, 1e-6)  # fmt: skip


def f16_vector8():
    """An eight-wide f16 vector load and store."""
    a0 = Var("a0")
    names = tuple(f"h{k}" for k in range(8))
    doubled = tuple(f"d{k}" for k in range(8))
    body = [
        Tile(
            axes=(ax("a0", 128),),
            body=(
                Load(names=names, input="h", index=(bx("*", a0, lit(8)),), dtype=F16),
                *(Assign(name=d, op="add", args=(n, n)) for n, d in zip(names, doubled, strict=True)),
                Write(output="out", index=(bx("*", a0, lit(8)),), values=doubled),
            ),
        )
    ]
    h = seeded([("h", (1024,), "f16")])["h"]
    package("p_syn_f16_vector8", "k_syn_f16v8", body, [("h", (1024,), "f16")], [("out", (1024,), "f16")], 1, 128,
            {"out": (h.astype(np.float32) * 2).astype(np.float16)}, 0, 0)  # fmt: skip


def aux_threads_and_barriers():
    """A Tile with 32 producer threads past the cells, plus warp and named barriers."""
    a0 = Var("a0")
    body = [
        Tile(
            axes=(ax("a0", 512),),
            block_threads=128,
            aux_threads=32,
            body=(
                Load(name="v", input="x", index=(a0,), dtype=F32),
                Sync(warp=True),
                Sync(barrier_id=1, count=160),
                Assign(name="w", op="multiply", args=("v", "v")),
                Write(output="out", index=(a0,), value="w"),
            ),
        )
    ]
    x = seeded([("x", (512,), "f32")])["x"]
    package("p_syn_aux_barriers", "k_syn_aux", body, [("x", (512,), "f32")], [("out", (512,), "f32")], 4, 160, {"out": x * x}, 0, 0)


def cp_async_small():
    """4- and 8-byte cp.async copies; each thread then reads another thread's data."""
    a0 = Var("a0")
    body = [
        Tile(
            axes=(ax("a0", 256),),
            body=(
                Smem(name="s", extents=(512,), dtype="float"),
                Smem(name="t", extents=(256,), dtype="float"),
                CpAsyncCopy(smem="s", smem_index=(bx("*", a0, lit(2)),), src="x", src_index=(bx("*", a0, lit(2)),), nbytes=8),
                CpAsyncCopy(smem="t", smem_index=(a0,), src="y", src_index=(a0,), nbytes=4),
                CpAsyncCommit(),
                CpAsyncWait(group=0),
                Sync(),
                Load(name="p", input="s", index=(bx("*", a0, lit(2)),), dtype=F32),
                Load(name="q", input="s", index=(bx("+", bx("*", a0, lit(2)), lit(1)),), dtype=F32),
                Load(name="r", input="t", index=(bx("-", lit(255), a0),), dtype=F32),
                Assign(name="pq", op="add", args=("p", "q")),
                Assign(name="o", op="add", args=("pq", "r")),
                Write(output="out", index=(a0,), value="o"),
            ),
        )
    ]
    ins = seeded([("x", (512,), "f32"), ("y", (256,), "f32")])
    x, y = ins["x"], ins["y"]
    package("p_syn_cp_async_4_8", "k_syn_cpasync", body, [("x", (512,), "f32"), ("y", (256,), "f32")], [("out", (256,), "f32")], 1, 256,
            {"out": (x[0::2] + x[1::2]) + y[::-1]}, 0, 0)  # fmt: skip


def segmented_tree():
    """A segmented TreeHalve: four rows per block, 32 threads per row; slot t*4+row."""
    rb, t, iv, j = Var("rb"), Var("t"), Var("iv"), Var("j")
    row = bx("+", bx("*", rb, lit(4)), iv)
    combine = (Assign(name="acc", op="add", args=("acc", "acc__o")),)
    body = [
        Tile(
            axes=(ax("rb", 16), ax("t", 32), ax("iv", 4)),
            block_threads=128,
            body=(
                StridedLoop(
                    axis=ax("j", 256),
                    start=t,
                    step=lit(32),
                    body=(
                        Load(name="v", input="x", index=(row, j), dtype=F32),
                        Accum(name="acc", value="v", op="add", dtype=F32, axes=("j",)),
                    ),
                ),
                Smem(name="buf", extents=(128,), dtype="float"),
                Write(output="buf", index=(bx("+", bx("*", t, lit(4)), iv),), value="acc"),
                Sync(),
                TreeHalve(
                    bufs=("buf",),
                    state=("acc",),
                    state_b=("acc__o",),
                    combine_states=combine,
                    length=32,
                    tid_var="t",
                    dtype=F32,
                    inner=("iv", 4),
                ),
                Cond(cond=bx("==", t, lit(0)), body=(Write(output="out", index=(row,), value="acc"),)),
            ),
        )
    ]
    x = seeded([("x", (64, 256), "f32")])["x"]
    package("r_syn_segmented_tree", "k_syn_tree", body, [("x", (64, 256), "f32")], [("out", (64,), "f32")], 16, 128,
            {"out": x.astype(np.float64).sum(-1).astype(np.float32)}, 1e-4, 1e-4)  # fmt: skip


def int64_shuffle_with_end():
    """A StridedLoop with an explicit end bound and an int64 warp-shuffle sum."""
    t, i = Var("t"), Var("i")
    body = [
        Tile(
            axes=(ax("t", 32),),
            block_threads=32,
            body=(
                StridedLoop(
                    axis=ax("i", 4096),
                    start=t,
                    step=lit(32),
                    end=lit(2000),
                    body=(Load(name="v", input="x", index=(i,), dtype=I64), Accum(name="acc", value="v", op="add", dtype=I64, axes=("i",))),
                ),
                WarpShuffle(
                    state=("acc",),
                    state_b=("acc__o",),
                    combine_states=(Assign(name="acc", op="add", args=("acc", "acc__o")),),
                    length=32,
                    dtype=I64,
                ),
                Cond(cond=bx("==", t, lit(0)), body=(Write(output="out", index=(), value="acc"),)),
            ),
        )
    ]
    x = seeded([("x", (4096,), "i64")])["x"]
    package("r_syn_int64_shuffle_end", "k_syn_i64", body, [("x", (4096,), "i64")], [("out", (1,), "i64")], 1, 32,
            {"out": np.array([x[:2000].sum()])}, 0, 0)  # fmt: skip


if __name__ == "__main__":
    for make in (cond_else_select_let, f16_vector8, aux_threads_and_barriers, cp_async_small, segmented_tree, int64_shuffle_with_end):
        make()
    print(f"wrote {len(list(OUT.iterdir()))} synthetic problems to {OUT}")
