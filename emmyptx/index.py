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

import math
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


def linearize(expr) -> tuple[dict, int]:
    """Write an integer expression as ``sum(coeff * atom) + constant``.

    Atoms are variables and the non-linear sub-expressions (``x / 64``,
    ``x % 3``, comparisons, ternaries), whose own operands are normalized
    recursively. Reordering +, - and * is exact in s32 arithmetic (it is
    arithmetic modulo 2^32), so the canonical form is always safe to use."""
    if isinstance(expr, Literal) and isinstance(expr.value, int):
        return {}, int(expr.value)
    if isinstance(expr, Var):
        return {expr: 1}, 0
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-"):
        lt, lc = linearize(expr.left)
        rt, rc = linearize(expr.right)
        sign = 1 if expr.op == "+" else -1
        terms = dict(lt)
        for atom, k in rt.items():
            terms[atom] = terms.get(atom, 0) + sign * k
        return {a: k for a, k in terms.items() if k}, lc + sign * rc
    if isinstance(expr, BinaryExpr) and expr.op == "*":
        lt, lc = linearize(expr.left)
        rt, rc = linearize(expr.right)
        if not lt or not rt:  # one side is a constant: scale the other
            terms, c, k = (rt, rc, lc) if not lt else (lt, lc, rc)
            return ({a: v * k for a, v in terms.items()} if k else {}), c * k
    if isinstance(expr, BinaryExpr):
        atom = BinaryExpr(expr.op, normalize(expr.left), normalize(expr.right))
    elif isinstance(expr, TernaryExpr):
        atom = TernaryExpr(normalize(expr.cond), normalize(expr.if_true), normalize(expr.if_false))
    else:
        atom = expr
    return {atom: 1}, 0


def rebuild(terms: dict, const: int = 0):
    """The canonical expression for ``sum(coeff * atom) + const`` (None if 0)."""
    out = None
    for atom, k in sorted(terms.items(), key=lambda t: repr(t[0])):
        term = atom if abs(k) == 1 else BinaryExpr("*", atom, Literal(abs(k), "int"))
        if out is None:
            out = term if k > 0 else BinaryExpr("-", Literal(0, "int"), term)
        else:
            out = BinaryExpr("+" if k > 0 else "-", out, term)
    if const:
        out = Literal(const, "int") if out is None else BinaryExpr("+", out, Literal(const, "int"))
    return out


def known_multiple(expr, var_multiples: dict[str, int] | None = None) -> int:
    """A number that provably divides the value of ``expr`` (0 means the value
    is 0). Used to prove alignment: ``(x * 4) % 64`` is always a multiple of 4.
    ``var_multiples`` gives what is known about variables (a loop counter that
    starts at 0 and steps by 64 is a multiple of 64)."""
    var_multiples = var_multiples or {}
    if isinstance(expr, Literal) and isinstance(expr.value, int):
        return abs(int(expr.value))
    if isinstance(expr, Var):
        return var_multiples.get(expr.name, 1)
    if isinstance(expr, BinaryExpr):
        a, b = known_multiple(expr.left, var_multiples), known_multiple(expr.right, var_multiples)
        if expr.op in ("+", "-", "%"):
            return math.gcd(a, b)
        if expr.op == "*":
            return a * b
    if isinstance(expr, TernaryExpr):
        return math.gcd(known_multiple(expr.if_true, var_multiples), known_multiple(expr.if_false, var_multiples))
    return 1


def normalize(expr):
    terms, const = linearize(expr)
    out = rebuild(terms, const)
    return Literal(0, "int") if out is None else out


def split_constant(expr) -> tuple[object | None, int]:
    """``(residual, constant)`` with ``expr == residual + constant``: the
    constant becomes an immediate address offset, so ``x[4*i + 1]`` and
    ``x[4*i + 2]`` share the address register of ``4*i``."""
    terms, const = linearize(expr)
    return rebuild(terms), const


def flatten(shape: tuple[int, ...], index: tuple):
    """Row-major flat element offset of ``index`` in a buffer of ``shape``:
    ``x[a, b]`` with shape (64, 512) is element ``a * 512 + b``."""
    if not index:
        return Literal(0, "int")
    if len(index) == 1:
        return index[0]
    if len(shape) != len(index):  # emmy's convention for a rank mismatch: sum the coordinates
        flat = index[0]
        for e in index[1:]:
            flat = BinaryExpr("+", flat, e)
        return flat
    flat = None
    for d, e in enumerate(index):
        stride = math.prod(shape[d + 1 :])
        term = e if stride == 1 else BinaryExpr("*", e, Literal(stride, "int"))
        flat = term if flat is None else BinaryExpr("+", flat, term)
    return flat


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
    def specialize(self, expr):
        """Replace names currently bound to constants (the counter of an
        unrolled loop) by literals, so constant folding can see them."""
        consts = {}
        for name in expr.free_vars():
            operand = self.resolve(name)
            if is_imm(operand):
                consts[name] = Literal(int(operand), "int")
        return expr.substitute(consts) if consts else expr

    def const(self, expr) -> int | None:
        """The compile-time value of ``expr``, or None."""
        if isinstance(expr, Var):
            operand = self.resolve(expr.name)
            return int(operand) if is_imm(operand) else None
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
        expr = normalize(expr)
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
        if is_imm(a) and is_imm(b) and (folded := _fold(expr.op, int(a), int(b))) is not None:
            return str(folded)
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
