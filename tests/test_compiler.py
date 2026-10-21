"""Offline tests: no GPU and no runner needed.

    python -m pytest tests

They check the PTX contract for every IR in the test suites (header, entry
name, parameter order, ASCII only, deterministic output) and the pieces of
the compiler that are easy to get subtly wrong (C integer semantics in
constant folding, affine normalization, f32 immediates). Numerical
correctness is checked on real hardware by tools/run_all.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

from emmyptx import compile_ir
from emmyptx.index import _fold, linearize, normalize, split_constant
from emmyptx.ptx import f32_imm

ROOT = Path(__file__).resolve().parent
IRS = sorted(ROOT.glob("open/*/*.ir.json")) + sorted(ROOT.glob("generated/*/*.ir.json"))


def lit(v):
    return Literal(v, "int")


@pytest.mark.parametrize("ir", IRS, ids=lambda p: p.parent.name)
def test_ptx_contract(ir: Path):
    ptx = compile_ir(ir)
    meta = json.loads((ir.parent / "meta.json").read_text())
    kernel = next(k for k in meta["kernels"] if f"{k['entry']}.ir.json" == ir.name)

    ptx.encode("ascii")  # ptxas rejects any non-ASCII byte
    directives = [line for line in ptx.splitlines() if line and not line.startswith("//")]
    assert directives[:3] == [".version 8.0", ".target sm_89", ".address_size 64"]
    assert f".visible .entry {kernel['entry']}(" in ptx
    params = re.findall(r"\.param \.u64 param_(\w+)", ptx)
    assert params == kernel["arg_order"]
    assert ptx == compile_ir(ir), "compilation must be deterministic"


def test_c_integer_semantics():
    assert _fold("/", 7, 2) == 3
    assert _fold("/", -7, 2) == -3  # C truncates toward zero
    assert _fold("%", -7, 2) == -1
    assert _fold("<", 1, 2) == 1
    assert _fold("/", 1, 0) is None  # never fold a division by zero


def test_affine_normalization_cancels_terms():
    a, b = Var("a"), Var("b")
    expr = BinaryExpr("-", BinaryExpr("+", BinaryExpr("*", a, lit(64)), BinaryExpr("*", b, lit(2))), BinaryExpr("*", a, lit(64)))
    assert normalize(expr) == BinaryExpr("*", b, lit(2))


def test_split_constant_peels_offsets():
    i = Var("i")
    residual, const = split_constant(BinaryExpr("*", BinaryExpr("+", i, lit(3)), lit(4)))
    assert residual == BinaryExpr("*", i, lit(4)) and const == 12


def test_nonlinear_atoms_are_kept():
    i = Var("i")
    terms, const = linearize(BinaryExpr("+", BinaryExpr("/", i, lit(8)), lit(1)))
    assert terms == {BinaryExpr("/", i, lit(8)): 1} and const == 1


def test_f32_immediates():
    assert f32_imm(1.0) == "0f3F800000"
    assert f32_imm(-1e30) == "0fF149F2CA"
