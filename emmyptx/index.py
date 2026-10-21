"""Integer and predicate expressions: indices, loop bounds and conditions.

IR index expressions are small trees of ``Var`` / ``Literal`` / ``BinaryExpr``
/ ``TernaryExpr`` nodes, e.g. ``((a0 * 4) + 1)``. This module turns such a
tree into s32 PTX instructions and returns an *operand*: either a register
name (``%r7``) or, when the whole tree is a compile-time constant, a decimal
immediate (``"12"``).

Two small optimisations keep the generated PTX short and readable:

* constant folding with C semantics (``7 / 2 == 3``, ``-7 % 2 == -1``) and the
  identities ``x + 0``, ``x * 1``, ``x * 0``;
* common-subexpression reuse: an expression already computed in the current
  (or an enclosing) block is not recomputed. Entries are dropped when a block
  closes (its registers no longer dominate later code) and when one of the
  names they read is reassigned.
"""

from __future__ import annotations

from collections.abc import Callable

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from emmyptx.loader import CompileError
from emmyptx.ptx import PtxBuilder

_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_ARITH = {
    "+": "add.s32",
    "-": "sub.s32",
    "*": "mul.lo.s32",
    "/": "div.s32",
    "//": "div.s32",
    "%": "rem.s32",
    "&": "and.b32",
    "|": "or.b32",
    "^": "xor.b32",
    "<<": "shl.b32",
    ">>": "shr.s32",
}
_COMMUTATIVE = {"+", "*", "&", "|", "^"}


def is_imm(operand: str) -> bool:
    return operand.lstrip("-").isdigit()


def _c_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _fold(op: str, a: int, b: int) -> int | None:
    if op in ("/", "//", "%") and b == 0:
        return None
    match op:
        case "+":
            return a + b
        case "-":
            return a - b
        case "*":
            return a * b
        case "/" | "//":
            return _c_div(a, b)
        case "%":
            return a - b * _c_div(a, b)
        case "&":
            return a & b
        case "|":
            return a | b
        case "^":
            return a ^ b
        case "<<":
            return a << b
        case ">>":
            return a >> b
        case "<":
            return int(a < b)
        case "<=":
            return int(a <= b)
        case ">":
            return int(a > b)
        case ">=":
            return int(a >= b)
        case "==":
            return int(a == b)
        case "!=":
            return int(a != b)
        case "&&":
            return int(bool(a) and bool(b))
        case "||":
            return int(bool(a) or bool(b))
    return None


def split_constant(expr) -> tuple[object | None, int]:
    """Split an affine expression into ``(residual, constant)`` so that
    ``expr == residual + constant``. ``x[(a0 * 4) + 3]`` then addresses as
    ``[base_of(a0 * 4) + 12]`` and the four unrolled loads share one register."""
    if isinstance(expr, Literal) and isinstance(expr.value, int):
        return None, int(expr.value)
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-"):
        left, cl = split_constant(expr.left)
        right, cr = split_constant(expr.right)
        if expr.op == "+":
            return _join("+", left, right), cl + cr
        return _join("-", left, right), cl - cr
    if isinstance(expr, BinaryExpr) and expr.op == "*":
        for lit, other in ((expr.left, expr.right), (expr.right, expr.left)):
            if isinstance(lit, Literal) and isinstance(lit.value, int):
                rest, c = split_constant(other)
                k = int(lit.value)
                return (None if rest is None else BinaryExpr("*", rest, Literal(k, "int"))), c * k
    return expr, 0


def _join(op: str, left, right):
    if right is None:
        return left
    if left is None:
        return right if op == "+" else BinaryExpr("-", Literal(0, "int"), right)
    return BinaryExpr(op, left, right)


class IndexLowerer:
    """Lowers integer / predicate expression trees to PTX operands."""

    def __init__(self, b: PtxBuilder, resolve: Callable[[str], str]):
        self.b = b
        self.resolve = resolve  # name -> s32 operand (register or immediate)
        self._scopes: list[dict] = [{}]

    # ---- scoping of the reuse cache -------------------------------------
    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        self._scopes.pop()

    def invalidate(self, name: str) -> None:
        """``name`` was (re)assigned: forget every cached value that read it."""
        for scope in self._scopes:
            for key in [k for k, (deps, _) in scope.items() if name in deps]:
                del scope[key]

    def cached(self, key, deps: frozenset[str], make: Callable[[], str]) -> str:
        for scope in reversed(self._scopes):
            if key in scope:
                return scope[key][1]
        value = make()
        self._scopes[-1][key] = (deps, value)
        return value

    # ---- integer values --------------------------------------------------
    def const(self, expr) -> int | None:
        """The compile-time value of ``expr``, or None."""
        if isinstance(expr, Literal):
            return int(expr.value) if isinstance(expr.value, (int, bool)) else None
        if isinstance(expr, BinaryExpr):
            a, b = self.const(expr.left), self.const(expr.right)
            return None if a is None or b is None else _fold(expr.op, a, b)
        if isinstance(expr, TernaryExpr):
            c = self.const(expr.cond)
            if c is not None:
                return self.const(expr.if_true if c else expr.if_false)
        return None

    def int_(self, expr) -> str:
        """Lower ``expr`` to an s32 operand (register or immediate)."""
        c = self.const(expr)
        if c is not None:
            return str(c)
        if isinstance(expr, Var):
            return self.resolve(expr.name)
        return self.cached(("int", expr), expr.free_vars(), lambda: self._int_uncached(expr))

    def _int_uncached(self, expr) -> str:
        if isinstance(expr, TernaryExpr):
            p = self.pred(expr.cond)
            t, f = self.int_(expr.if_true), self.int_(expr.if_false)
            r = self.b.reg("s32")
            self.b.ins("selp.s32", r, t, f, p)
            return r
        if not isinstance(expr, BinaryExpr):
            raise CompileError(f"unsupported index expression {type(expr).__name__}: {expr!r}")
        if expr.op in _CMP or expr.op in ("&&", "||"):
            r = self.b.reg("s32")
            self.b.ins("selp.s32", r, "1", "0", self.pred(expr))
            return r
        if expr.op not in _ARITH:
            raise CompileError(f"unsupported index operator {expr.op!r}")
        a, b = self.int_(expr.left), self.int_(expr.right)
        simplified = _simplify(expr.op, a, b)
        if simplified is not None:
            return simplified
        if is_imm(a):
            if expr.op in _COMMUTATIVE:
                a, b = b, a
            else:
                a = self._materialize(a)
        r = self.b.reg("s32")
        self.b.ins(_ARITH[expr.op], r, a, b)
        return r

    def _materialize(self, imm: str) -> str:
        r = self.b.reg("s32")
        self.b.ins("mov.s32", r, imm)
        return r

    # ---- predicates ------------------------------------------------------
    def pred(self, expr) -> str:
        """Lower ``expr`` to a predicate register (non-zero means true)."""
        return self.cached(("pred", expr), expr.free_vars(), lambda: self._pred_uncached(expr))

    def _pred_uncached(self, expr) -> str:
        p = self.b.reg("pred")
        if isinstance(expr, BinaryExpr) and expr.op in _CMP:
            a, b = self.int_(expr.left), self.int_(expr.right)
            if is_imm(a):
                a = self._materialize(a)
            self.b.ins(f"setp.{_CMP[expr.op]}.s32", p, a, b)
        elif isinstance(expr, BinaryExpr) and expr.op in ("&&", "||"):
            pa, pb = self.pred(expr.left), self.pred(expr.right)
            self.b.ins("and.pred" if expr.op == "&&" else "or.pred", p, pa, pb)
        else:
            v = self.int_(expr)
            if is_imm(v):
                v = self._materialize(v)
            self.b.ins("setp.ne.s32", p, v, "0")
        return p


def _simplify(op: str, a: str, b: str) -> str | None:
    """Algebraic identities that make an instruction unnecessary."""
    if op == "+" and b == "0" or op in ("-", "<<", ">>") and b == "0":
        return a
    if op == "+" and a == "0":
        return b
    if op == "*" and (a == "0" or b == "0"):
        return "0"
    if op == "*" and b == "1" or op in ("/", "//") and b == "1":
        return a
    if op == "*" and a == "1":
        return b
    if op == "%" and b == "1":
        return "0"
    return None
