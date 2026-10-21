#!/usr/bin/env python3
"""Compile Emmy Kernel IR to PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

The launch contract (parameter order, grid, block, shared memory) is read
from the meta.json next to the IR file.
"""

import sys

from emmyptx import CompileError, compile_ir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_ir(sys.argv[1])
    except CompileError as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
