"""Stage 1: load a Kernel IR file and its launch contract.

The IR json holds a small graph: input / constant nodes that describe global
buffers (shape and dtype), and exactly one ``KernelOp`` whose ``body`` is the
statement tree we compile. ``Graph.from_dict`` (emmy's own loader) turns the
body into typed statement objects; we never parse it ourselves.

The kernel's parameter order and launch geometry come from ``meta.json`` in
the same folder: that file is the contract the runner launches with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from emmy.compiler.graph import Graph

from emmyptx import dtypes
from emmyptx.dtypes import DType
from emmyptx.errors import CompileError


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: DType

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Launch:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_smem: int  # bytes of dynamic shared memory passed at launch

    @property
    def threads(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2] * self.block[0] * self.block[1] * self.block[2]


@dataclass(frozen=True)
class KernelSpec:
    """Everything the lowering needs to know about one kernel."""

    name: str
    body: tuple  # emmy statement objects (the KernelOp body)
    params: tuple[str, ...]  # buffer names in launch argument order
    buffers: dict[str, Buffer]  # every global buffer the kernel can touch
    outputs: frozenset[str]  # buffers the kernel writes
    launch: Launch | None  # None when no meta.json is available


def load_kernel(ir_path: str | Path) -> KernelSpec:
    ir_path = Path(ir_path)
    graph_json = json.loads(ir_path.read_text())
    graph = Graph.from_dict(graph_json)

    kernels = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if len(kernels) != 1:
        raise CompileError(f"expected exactly one KernelOp in {ir_path.name}, found {len(kernels)}")
    node = kernels[0]
    op = node.op
    if op.serial:
        raise CompileError(f"serial axes {op.serial} (runtime coordinate parameters) are not supported")

    buffers: dict[str, Buffer] = {}
    for n in graph.nodes.values():
        for t in n.outputs:
            shape = tuple(int(d.as_static()) for d in t.shape) or (1,)
            buffers[t.name] = Buffer(t.name, shape, dtypes.get(t.dtype))
    outputs = frozenset(t.name for t in node.outputs)

    meta_kernel = _find_meta(ir_path, op.name)
    if meta_kernel is not None:
        params = tuple(meta_kernel["arg_order"])
        launch = Launch(
            grid=tuple(meta_kernel["grid"]),
            block=tuple(meta_kernel["block"]),
            dynamic_smem=int(meta_kernel.get("smem_bytes", 0)),
        )
    else:
        # Without a launch contract fall back to emmy's own convention:
        # inputs in first-use order, then outputs.
        params = tuple(dict.fromkeys([*node.inputs, *(t.name for t in node.outputs)]))
        launch = None
    for p in params:
        if p not in buffers:
            raise CompileError(f"parameter {p!r} names no buffer in the IR graph")

    return KernelSpec(
        name=op.name,
        body=tuple(op.body),
        params=params,
        buffers=buffers,
        outputs=outputs,
        launch=launch,
    )


def _find_meta(ir_path: Path, entry: str) -> dict | None:
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    for k in meta.get("kernels", []):
        if k.get("entry") == entry:
            return k
    raise CompileError(f"meta.json has no kernel with entry {entry!r}")
