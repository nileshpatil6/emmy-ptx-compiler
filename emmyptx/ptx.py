"""A small PTX text builder: virtual registers, labels and instruction lines.

The builder knows nothing about Kernel IR. It hands out fresh virtual
registers per register class (``%f1``, ``%f2``, ...), fresh labels, and
collects instruction lines; ``render_kernel`` then wraps them in the module
header, the entry signature and the ``.reg`` declarations. ``ptxas`` performs
the real register allocation, so every value simply gets a new register.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"

# register class -> (name prefix, .reg declaration type)
REG_CLASSES: dict[str, tuple[str, str]] = {
    "pred": ("%p", ".pred"),
    "b16": ("%h", ".b16"),
    "f32": ("%f", ".f32"),
    "f64": ("%fd", ".f64"),
    "s32": ("%r", ".b32"),
    "s64": ("%rd", ".b64"),
}


@dataclass
class SharedArray:
    """One ``.shared`` declaration at module scope."""

    name: str
    nbytes: int
    align: int


@dataclass
class PtxBuilder:
    lines: list[str] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in REG_CLASSES})
    _labels: int = 0

    def reg(self, cls: str) -> str:
        """A fresh virtual register of class ``cls`` (see ``REG_CLASSES``)."""
        self._counts[cls] += 1
        return f"{REG_CLASSES[cls][0]}{self._counts[cls]}"

    def label(self, hint: str) -> str:
        self._labels += 1
        return f"$L_{hint}_{self._labels}"

    def ins(self, opcode: str, *operands: str, pred: str | None = None) -> None:
        """Emit one instruction: ``[@pred] opcode a, b, c;``."""
        guard = f"@{pred} " if pred else ""
        text = f"{guard}{opcode}"
        if operands:
            text = f"{text:<24} {', '.join(operands)}"
        self.lines.append(f"\t{text};")

    def place(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def branch(self, label: str, pred: str | None = None) -> None:
        self.ins("bra", label, pred=pred)

    def comment(self, text: str) -> None:
        # PTX rejects non-ASCII anywhere, comments included.
        safe = text.encode("ascii", "replace").decode("ascii")
        self.lines.append(f"\t// {safe}")

    def reg_decls(self) -> list[str]:
        out = []
        for cls, n in self._counts.items():
            if n:
                prefix, ty = REG_CLASSES[cls]
                out.append(f"\t.reg {ty:<6} {prefix}<{n + 1}>;")
        return out


def f32_imm(value: float) -> str:
    """An f32 immediate in PTX's exact hex form (``0f3F800000`` is 1.0)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


def f64_imm(value: float) -> str:
    return "0d%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def render_kernel(
    name: str,
    params: list[str],
    body: PtxBuilder,
    shared: list[SharedArray],
    dynamic_shared: SharedArray | None,
    max_threads: int | None,
    header_comment: list[str],
) -> str:
    """Assemble the complete PTX module for one kernel entry."""
    out = [f"// {line}" for line in header_comment]
    out += [f".version {PTX_VERSION}", f".target {PTX_TARGET}", ".address_size 64", ""]
    for s in shared:
        out.append(f".shared .align {s.align} .b8 {s.name}[{s.nbytes}];")
    if dynamic_shared is not None:
        out.append(f".extern .shared .align {dynamic_shared.align} .b8 {dynamic_shared.name}[];")
    if shared or dynamic_shared:
        out.append("")
    sig = ",\n".join(f"\t.param .u64 {p}" for p in params)
    out.append(f".visible .entry {name}(\n{sig}\n)")
    if max_threads:
        out.append(f".maxntid {max_threads}, 1, 1")
    out.append("{")
    out += body.reg_decls()
    out.append("")
    out += body.lines
    out.append("}")
    text = "\n".join(out) + "\n"
    text.encode("ascii")  # fail loudly here rather than in ptxas
    return text
