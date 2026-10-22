"""emmyptx: a compiler from Emmy Kernel IR (json) to PTX.

Pipeline:  loader.load_kernel  ->  lower.lower_kernel  ->  PTX text
"""

from __future__ import annotations

from pathlib import Path

from emmyptx.errors import CompileError
from emmyptx.loader import load_kernel
from emmyptx.lower import lower_kernel

__all__ = ["CompileError", "compile_ir"]


def compile_ir(ir_path: str | Path) -> str:
    """Compile one ``*.ir.json`` file (with its ``meta.json`` alongside) to PTX."""
    return lower_kernel(load_kernel(ir_path))
