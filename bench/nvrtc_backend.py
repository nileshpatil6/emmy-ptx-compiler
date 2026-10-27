"""Emmy CUDA renderer and CUDA 12 NVRTC, confined to the comparison tool."""

from __future__ import annotations

import ctypes as ct
import json
import os
import re
import sys
import time
from pathlib import Path

OPTIONS = ("--gpu-architecture=compute_89", "--std=c++17", "--fmad=false")


def render_cuda(ir: Path):
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.kernel.render import render_kernelop

    graph = Graph.from_dict(json.loads(ir.read_text()))
    nodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one kernel")
    op = nodes[0].op
    meta = json.loads((ir.parent / "meta.json").read_text())
    kernel = next(k for k in meta["kernels"] if k["entry"] == op.name)
    tensors = {t.name: t for n in graph.nodes.values() for t in n.outputs}
    source = render_kernelop(op, tensors=tensors)
    # CUDA lowering may order buffers differently from the body's first use.
    # Reorder only declarations to honor the exact meta.json positional ABI.
    signature = re.search(r"void " + re.escape(op.name) + r"\(([^)]*)\)", source)
    if signature is None:
        raise ValueError("Cannot identify rendered CUDA signature")
    declarations = signature.group(1).split(",")
    params = {re.search(r"(\w+)\s*$", declaration).group(1): declaration.strip() for declaration in declarations}
    if set(params) != set(kernel["arg_order"]) or len(params) != len(declarations):
        raise ValueError("Rendered CUDA buffers differ from launch arguments")
    ordered = ", ".join(params[name] for name in kernel["arg_order"])
    return source[:signature.start(1)] + ordered + source[signature.end(1):]


class NVRTC:
    def __init__(self):
        import nvidia.cuda_nvrtc

        base = Path(nvidia.cuda_nvrtc.__file__).parent
        paths = list(base.rglob("nvrtc64_*.dll")) if sys.platform == "win32" else list(base.rglob("libnvrtc.so*"))
        paths = [p for p in paths if "builtins" not in p.name and ".alt." not in p.name]
        if not paths:
            raise RuntimeError("NVRTC library missing; install nvidia-cuda-nvrtc-cu12")
        self.dll_dir = os.add_dll_directory(str(paths[0].parent)) if sys.platform == "win32" else None
        # NVRTC loads builtins by basename; preload the wheel's matching DLL.
        self.builtins = [ct.CDLL(str(p)) for p in base.rglob("nvrtc-builtins64_*.dll")] if sys.platform == "win32" else []
        self.lib = ct.CDLL(str(paths[0]))
        self.lib.nvrtcCreateProgram.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int,
                                               ct.POINTER(ct.c_char_p), ct.POINTER(ct.c_char_p)]
        self.lib.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
        self.lib.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
        for kind in ("PTX", "ProgramLog"):
            getattr(self.lib, f"nvrtcGet{kind}Size").argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
            getattr(self.lib, f"nvrtcGet{kind}").argtypes = [ct.c_void_p, ct.c_void_p]
        major, minor = ct.c_int(), ct.c_int()
        self.lib.nvrtcVersion(ct.byref(major), ct.byref(minor))
        self.version = f"{major.value}.{minor.value}"

    def _text(self, program, kind):
        size = ct.c_size_t()
        if getattr(self.lib, f"nvrtcGet{kind}Size")(program, ct.byref(size)):
            raise RuntimeError(f"Cannot get NVRTC {kind} size")
        buf = ct.create_string_buffer(size.value)
        if getattr(self.lib, f"nvrtcGet{kind}")(program, buf):
            raise RuntimeError(f"Cannot retrieve NVRTC {kind}")
        return buf.value.decode()

    def compile(self, source):
        program = ct.c_void_p()
        status = self.lib.nvrtcCreateProgram(ct.byref(program), source.encode(), b"kernel.cu", 0, None, None)
        if status:
            raise RuntimeError(f"nvrtcCreateProgram: {status}")
        try:
            options = (ct.c_char_p * len(OPTIONS))(*(x.encode() for x in OPTIONS))
            start = time.perf_counter()
            status = self.lib.nvrtcCompileProgram(program, len(options), options)
            ms = (time.perf_counter() - start) * 1000
            if status:
                raise RuntimeError(self._text(program, "ProgramLog"))
            return self._text(program, "PTX"), ms
        finally:
            self.lib.nvrtcDestroyProgram(ct.byref(program))


if __name__ == "__main__":
    print(NVRTC().compile(render_cuda(Path(sys.argv[1])))[0], end="")
