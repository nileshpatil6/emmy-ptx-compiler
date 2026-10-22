"""Scalar values and the elementwise operations of ``Assign`` / ``Accum``.

A ``Value`` is a PTX operand plus the dtype it holds. ``Elementwise`` emits
the instruction sequence for one named op (``add``, ``exp``, ``tanh``, ...)
following emmy's typing: integer ops stay integer, everything else is
computed in f32 and converted to the result dtype once.

PTX has no ``exp`` or ``tanh`` instruction. It has fast hardware
approximations of 2^x (``ex2.approx``), log2 (``lg2.approx``), sin and cos;
the transcendental functions below are built from those with the usual
identities (exp(x) = 2^(x * log2 e), ...), with care where an identity would
lose relative precision (tanh and erf near zero use their Taylor series).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from emmyptx import dtypes
from emmyptx.dtypes import BOOL, F16, F32, F64, DType
from emmyptx.errors import CompileError
from emmyptx.ptx import PtxBuilder, f32_imm


@dataclass(frozen=True)
class Value:
    reg: str  # register name, or an immediate for integer constants
    dtype: DType


_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_LN2_HI = struct.unpack("<f", struct.pack("<f", _LN2))[0]  # ln 2 rounded to f32 ...
_LN2_LO = _LN2 - _LN2_HI  # ... and the part it misses


def _pio2_parts() -> tuple[float, float, float]:
    """pi/2 as a sum of three f32 values (each one the rounding of what remains)."""
    parts, rest = [], math.pi / 2
    for _ in range(3):
        part = struct.unpack("<f", struct.pack("<f", rest))[0]
        parts.append(part)
        rest -= part
    return tuple(parts)


_PIO2_PARTS = _pio2_parts()

_FLOAT_COMPARE = {"less": "lt", "less_equal": "le", "greater": "gt", "greater_equal": "ge", "equal": "eq", "not_equal": "neu"}


# Integer results of these ops are computed exactly in integer registers; any
# other op with an integer result is computed in f32 and converted, which is
# what emmy's C does (``long long v = fmaxf(...)``).
_EXACT_INTEGER_OPS = dtypes.INTEGER_OPS | {"maximum", "minimum", "copy"}


class Elementwise:
    def __init__(self, b: PtxBuilder):
        self.b = b
        # Inside loops (online softmax, attention) exp runs once per element of
        # a reduction whose tolerance already absorbs reordered sums; there the
        # two-instruction exp is used. Straight-line code gets the libm-grade one.
        self.loop_depth = 0

    # ---- conversions -----------------------------------------------------
    def convert(self, v: Value, to: DType) -> Value:
        """Convert ``v`` to dtype ``to`` with C semantics (float -> int truncates,
        anything -> bool tests for non-zero)."""
        src = v.dtype
        if to is BOOL and src is not BOOL:
            return self._to_bool(v)
        if src is to or (not src.is_float and not to.is_float and src.reg == to.reg and max(src.size, to.size) <= 4):
            return Value(v.reg, to)  # same 32-bit register, nothing to do
        if F16 in (src, to) and F32 not in (src, to):
            return self.convert(self.convert(v, F32), to)  # f16 converts via f32
        r = self.b.reg(to.reg)
        if src.is_float and to.is_float:
            mode = "" if to.size > src.size else ".rn"  # widening is exact
            self.b.ins(f"cvt{mode}.{to.mem}.{src.mem}", r, v.reg)
        elif to.is_float:
            self.b.ins(f"cvt.rn.{to.mem}.{_int_cvt(src)}", r, self._reg(v))
        elif src.is_float:
            self.b.ins(f"cvt.rzi.{_int_cvt(to)}.{src.mem}", r, v.reg)
        else:  # integer -> integer of a different register width
            self.b.ins(f"cvt.{_int_cvt(to)}.{_int_cvt(src)}", r, self._reg(v))
        return Value(r, to)

    def _to_bool(self, v: Value) -> Value:
        p, r = self.b.reg("pred"), self.b.reg("s32")
        if v.dtype.is_float:
            v = self.convert(v, F32)
            self.b.ins("setp.ne.f32", p, v.reg, f32_imm(0.0))
        else:
            self.b.ins(f"setp.ne.{_int_cvt(v.dtype)}", p, self._reg(v), "0")
        self.b.ins("selp.u32", r, "1", "0", p)
        return Value(r, BOOL)

    def _reg(self, v: Value) -> str:
        """Integer immediates must live in a register for some instructions."""
        if v.reg.startswith("%"):
            return v.reg
        r = self.b.reg(v.dtype.reg)
        self.b.ins(f"mov.{_int_cvt(v.dtype)}", r, v.reg)
        return r

    def constant(self, value: float, dtype: DType) -> Value:
        """Materialize a numeric constant (an accumulator identity, a literal)."""
        if not dtype.is_float:  # saturate like cvt.rzi (the max identity is -1e30)
            bits = dtype.size * 8
            lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if dtype.kind == "int" else (0, 2**bits - 1)
            return Value(str(min(max(int(value), lo), hi)), dtype)
        r = self.b.reg(F32.reg)
        self.b.ins("mov.f32", r, f32_imm(value))
        return self.convert(Value(r, F32), dtype)

    def fresh(self, v: Value) -> Value:
        """A copy of ``v`` in a new register (accumulators must be registers)."""
        r = self.b.reg(v.dtype.reg)
        self.b.ins(f"mov.{_mov_type(v.dtype)}", r, v.reg)
        return Value(r, v.dtype)

    def move(self, dst: Value, src: Value) -> None:
        """Copy ``src`` into the existing register ``dst`` (an in-place update)."""
        src = self.convert(src, dst.dtype)
        self.b.ins(f"mov.{_mov_type(dst.dtype)}", dst.reg, src.reg)

    # ---- one elementwise op ----------------------------------------------
    def apply(self, op: str, args: list[Value], result: DType | None = None, into: Value | None = None) -> Value:
        """``result = op(args)``; ``result`` defaults to emmy's promotion rule.
        With ``into`` (an existing f32 register, e.g. an accumulator) a
        single-instruction op writes its result there directly."""
        result = result or dtypes.promote(op, [a.dtype for a in args])
        if into is not None and into.dtype is F32 and result is F32:
            reg = self._f32(op, [self.convert(a, F32).reg for a in args], out=into.reg)
            if reg != into.reg:
                self.b.ins("mov.f32", into.reg, reg)
            return into
        if result.is_int and all(a.dtype.is_int for a in args) and op in _EXACT_INTEGER_OPS:
            return self._integer(op, [self.convert(a, result) for a in args], result)
        fargs = [self.convert(a, F32).reg for a in args]
        return self.convert(Value(self._f32(op, fargs), F32), result)

    def _integer(self, op: str, args: list[Value], dt: DType) -> Value:
        t = _int_cvt(dt)
        bits = f"b{dt.size * 8 if dt.size > 4 else 32}"
        r = self.b.reg(dt.reg)
        a = [self._reg(args[0]), *(x.reg for x in args[1:])]
        match op:
            case "add" | "subtract":
                self.b.ins(f"{'add' if op == 'add' else 'sub'}.{t}", r, *a)
            case "multiply":
                self.b.ins(f"mul.lo.{t}", r, *a)
            case "floor_divide":  # emmy renders integer // as C '/', which truncates
                self.b.ins(f"div.{t}", r, *a)
            case "remainder" | "mod":
                self.b.ins(f"rem.{t}", r, *a)
            case "left_shift":
                self.b.ins(f"shl.{bits}", r, a[0], self.convert(args[1], dtypes.U32).reg)
            case "right_shift":
                self.b.ins(f"shr.{t}", r, a[0], self.convert(args[1], dtypes.U32).reg)
            case "bitwise_and" | "bitwise_or" | "bitwise_xor":
                self.b.ins(f"{op.removeprefix('bitwise_')}.{bits}", r, *a)
            case "maximum" | "minimum":
                self.b.ins(f"{op[:3]}.{t}", r, *a)
            case "copy":
                self.b.ins(f"mov.{bits}", r, a[0])
            case "bitwise_count":
                c = self.b.reg("s32")
                self.b.ins(f"popc.{bits}", c, a[0])
                return self.convert(Value(c, dtypes.U32), dt)
            case _:
                raise CompileError(f"integer op {op!r} is not supported")
        return Value(r, dt)

    def _f32(self, op: str, a: list[str], out: str | None = None) -> str:
        b = self.b
        r = out or b.reg("f32")
        match op:
            case "add" | "subtract" | "multiply" | "divide":
                # .rn: correctly rounded, and never contracted into an fma, so
                # pointwise results match a numpy reference bit for bit.
                name = {"add": "add", "subtract": "sub", "multiply": "mul", "divide": "div"}[op]
                b.ins(f"{name}.rn.f32", r, a[0], a[1])
            case "square":
                b.ins("mul.rn.f32", r, a[0], a[0])
            case "maximum" | "minimum":
                b.ins(f"{op[:3]}.f32", r, a[0], a[1])
            case "relu":
                b.ins("max.f32", r, a[0], f32_imm(0.0))
            case "negative":
                b.ins("neg.f32", r, a[0])
            case "abs":
                b.ins("abs.f32", r, a[0])
            case "copy" | "pad":  # a real copy: the source may be an accumulator updated later
                b.ins("mov.f32", r, a[0])
            case "reciprocal":
                b.ins("rcp.rn.f32", r, a[0])
            case "sqrt":
                b.ins("sqrt.rn.f32", r, a[0])
            case "rsqrt":
                b.ins("rsqrt.approx.f32", r, a[0])
            case "exp" | "exp_fast":
                return self._exp_fast(a[0]) if self.loop_depth or op == "exp_fast" else self._exp(a[0])
            case "log":
                b.ins("lg2.approx.f32", r, a[0])
                return self._mul_imm(r, _LN2)
            case "sin" | "cos":
                return self._sincos(op, a[0])
            case "tanh":
                return self._tanh(a[0])
            case "erf":
                return self._erf(a[0])
            case "sigmoid":  # 1 / (1 + exp(-x)), as emmy spells it
                neg = b.reg("f32")
                b.ins("neg.f32", neg, a[0])
                e = self._exp(neg)
                b.ins("add.rn.f32", r, e, f32_imm(1.0))
                out = b.reg("f32")
                b.ins("rcp.rn.f32", out, r)
                return out
            case "pow":
                return self._pow(a[0], a[1])
            case "where":
                p = b.reg("pred")
                b.ins("setp.ne.f32", p, a[0], f32_imm(0.0))
                b.ins("selp.f32", r, a[1], a[2], p)
            case _ if op in _FLOAT_COMPARE:
                p = b.reg("pred")
                b.ins(f"setp.{_FLOAT_COMPARE[op]}.f32", p, a[0], a[1])
                b.ins("selp.f32", r, f32_imm(1.0), f32_imm(0.0), p)
            case "bitwise_and" | "bitwise_or":  # logical and / or of float masks
                pa, pb, p = b.reg("pred"), b.reg("pred"), b.reg("pred")
                b.ins("setp.ne.f32", pa, a[0], f32_imm(0.0))
                b.ins("setp.ne.f32", pb, a[1], f32_imm(0.0))
                b.ins("and.pred" if op == "bitwise_and" else "or.pred", p, pa, pb)
                b.ins("selp.f32", r, f32_imm(1.0), f32_imm(0.0), p)
            case "bitwise_not":  # logical not of a float mask
                p = b.reg("pred")
                b.ins("setp.eq.f32", p, a[0], f32_imm(0.0))
                b.ins("selp.f32", r, f32_imm(1.0), f32_imm(0.0), p)
            case _:
                raise CompileError(f"elementwise op {op!r} is not supported")
        return r

    # ---- transcendental helpers (all f32) --------------------------------
    def _mul_imm(self, x: str, k: float) -> str:
        r = self.b.reg("f32")
        self.b.ins("mul.rn.f32", r, x, f32_imm(k))
        return r

    def _fma_imm(self, x: str, k: float, c: str) -> str:
        """``x * k + c`` with ``c`` a register or immediate."""
        r = self.b.reg("f32")
        self.b.ins("fma.rn.f32", r, x, f32_imm(k), c)
        return r

    def _poly(self, x: str, coeffs: list[float]) -> str:
        """Horner evaluation of ``sum(coeffs[i] * x**i)``."""
        acc = self.b.reg("f32")
        self.b.ins("mov.f32", acc, f32_imm(coeffs[-1]))
        for c in reversed(coeffs[:-1]):
            nxt = self.b.reg("f32")
            self.b.ins("fma.rn.f32", nxt, acc, x, f32_imm(c))
            acc = nxt
        return acc

    def _exp_fast(self, x: str) -> str:
        """exp(x) = 2^(x * log2 e) in two instructions, CUDA's __expf sequence
        (relative error ~2e-7 plus ~4e-8 per unit of |x|)."""
        r = self.b.reg("f32")
        self.b.ins("ex2.approx.f32", r, self._mul_imm(x, _LOG2E))
        return r

    def _exp(self, x: str) -> str:
        """exp(x) = 2^n * exp(r) with n = round(x / ln 2) and r = x - n ln 2
        computed exactly (Cody-Waite, two fmas), so |r| <= ln(2) / 2 and the
        hardware 2^x only sees a small argument; 2^n is an exact power of two
        (split in two factors so n = 128 does not overflow early). Accurate to
        a few ulp over the whole range, like libm's expf."""
        b = self.b
        xc = b.reg("f32")  # clamp: below -104 the result is 0, above 89 it is inf (NaN stays NaN)
        b.ins("max.NaN.f32", xc, x, f32_imm(-104.0))
        b.ins("min.NaN.f32", xc, xc, f32_imm(89.0))
        n = b.reg("f32")
        b.ins("mul.rn.f32", n, xc, f32_imm(_LOG2E))
        b.ins("cvt.rni.f32.f32", n, n)
        r = self._fma_imm(n, -_LN2_HI, xc)
        r = self._fma_imm(n, -_LN2_LO, r)
        p = b.reg("f32")
        b.ins("ex2.approx.f32", p, self._mul_imm(r, _LOG2E))
        n1 = self._mul_imm(n, 0.5)
        b.ins("cvt.rmi.f32.f32", n1, n1)
        n2 = b.reg("f32")
        b.ins("sub.rn.f32", n2, n, n1)
        for half in (n1, n2):
            scale = b.reg("f32")
            b.ins("ex2.approx.f32", scale, half)  # exact for an integer argument
            nxt = b.reg("f32")
            b.ins("mul.rn.f32", nxt, p, scale)
            p = nxt
        return p

    def _sincos(self, op: str, x: str) -> str:
        """Reduce x by multiples of pi/2 (three-part Cody-Waite), evaluate the
        sin or cos polynomial on [-pi/4, pi/4], then pick the quadrant's
        function and sign: cos(x) is sin one quadrant later."""
        b = self.b
        j = self._mul_imm(x, 2 / math.pi)
        b.ins("cvt.rni.f32.f32", j, j)
        r = x
        for part in _PIO2_PARTS:
            r = self._fma_imm(j, -part, r)
        r2 = b.reg("f32")
        b.ins("mul.rn.f32", r2, r, r)
        sin_poly = self._poly(r2, [0.0, -1.6666654611e-1, 8.3321608736e-3, -1.9515295891e-4])
        sin_r = b.reg("f32")
        b.ins("fma.rn.f32", sin_r, sin_poly, r, r)  # r + r * (r2 * poly)
        cos_poly = self._poly(r2, [1.0, -0.5, 4.166664568298827e-2, -1.388731625493765e-3, 2.443315711809948e-5])
        q = b.reg("s32")
        b.ins("cvt.rzi.s32.f32", q, j)
        if op == "cos":
            b.ins("add.s32", q, q, "1")
        odd, neg = b.reg("s32"), b.reg("s32")
        b.ins("and.b32", odd, q, "1")
        b.ins("and.b32", neg, q, "2")
        p_odd, p_neg = b.reg("pred"), b.reg("pred")
        b.ins("setp.ne.s32", p_odd, odd, "0")
        b.ins("setp.ne.s32", p_neg, neg, "0")
        v = b.reg("f32")
        b.ins("selp.f32", v, cos_poly, sin_r, p_odd)
        nv = b.reg("f32")
        b.ins("neg.f32", nv, v)
        out = b.reg("f32")
        b.ins("selp.f32", out, nv, v, p_neg)
        return out

    def _tanh(self, x: str) -> str:
        """|x| < 0.6: odd Taylor polynomial (keeps relative precision near 0);
        otherwise sign(x) * (1 - 2 / (exp(2|x|) + 1))."""
        b = self.b
        ax = b.reg("f32")
        b.ins("abs.f32", ax, x)
        x2 = b.reg("f32")
        b.ins("mul.rn.f32", x2, x, x)
        series = self._poly(x2, [1.0, -1 / 3, 2 / 15, -17 / 315, 62 / 2835, -1382 / 155925, 21844 / 6081075, -929569 / 638512875])
        small = b.reg("f32")
        b.ins("mul.rn.f32", small, series, x)
        e = self._exp(self._mul_imm(ax, 2.0))
        d = b.reg("f32")
        b.ins("add.rn.f32", d, e, f32_imm(1.0))
        q = b.reg("f32")
        b.ins("rcp.rn.f32", q, d)
        big = self._fma_imm(q, -2.0, f32_imm(1.0))
        signed = b.reg("f32")
        b.ins("copysign.f32", signed, x, big)
        p = b.reg("pred")
        b.ins("setp.lt.f32", p, ax, f32_imm(0.6))
        r = b.reg("f32")
        b.ins("selp.f32", r, small, signed, p)
        return r

    def _erf(self, x: str) -> str:
        """|x| < 0.5: Taylor series; otherwise Abramowitz-Stegun 7.1.26
        (absolute error below 1.5e-7) with the sign restored."""
        b = self.b
        ax = b.reg("f32")
        b.ins("abs.f32", ax, x)
        x2 = b.reg("f32")
        b.ins("mul.rn.f32", x2, x, x)
        c = 2 / math.sqrt(math.pi)
        series = self._poly(x2, [c, -c / 3, c / 10, -c / 42, c / 216, -c / 1320])
        small = b.reg("f32")
        b.ins("mul.rn.f32", small, series, x)
        t_den = self._fma_imm(ax, 0.3275911, f32_imm(1.0))
        t = b.reg("f32")
        b.ins("rcp.rn.f32", t, t_den)
        poly = self._poly(t, [0.0, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])
        nx2 = b.reg("f32")
        b.ins("neg.f32", nx2, x2)
        e = self._exp(nx2)
        tail = b.reg("f32")
        b.ins("mul.rn.f32", tail, poly, e)
        big = b.reg("f32")
        b.ins("sub.rn.f32", big, f32_imm(1.0), tail)
        signed = b.reg("f32")
        b.ins("copysign.f32", signed, x, big)
        p = b.reg("pred")
        b.ins("setp.lt.f32", p, ax, f32_imm(0.5))
        r = b.reg("f32")
        b.ins("selp.f32", r, small, signed, p)
        return r

    def _pow(self, base: str, expo: str) -> str:
        """base^expo = 2^(expo * log2|base|); a negative base takes the sign of
        an odd integer exponent (NaN for a non-integer one, as in C)."""
        b = self.b
        ab = b.reg("f32")
        b.ins("abs.f32", ab, base)
        lg = b.reg("f32")
        b.ins("lg2.approx.f32", lg, ab)
        t = b.reg("f32")
        b.ins("mul.rn.f32", t, expo, lg)
        mag = b.reg("f32")
        b.ins("ex2.approx.f32", mag, t)
        # expo == 0 gives 1 even for base 0 (0 * -inf would be NaN)
        one = b.reg("pred")
        b.ins("setp.eq.f32", one, expo, f32_imm(0.0))
        b.ins("selp.f32", mag, f32_imm(1.0), mag, one)
        # sign for a negative base: odd integer exponent -> negative, non-integer -> NaN
        half = self._mul_imm(expo, 0.5)
        fl = b.reg("f32")
        b.ins("cvt.rmi.f32.f32", fl, half)
        odd = b.reg("pred")
        b.ins("setp.ne.f32", odd, fl, half)
        neg_mag = b.reg("f32")
        b.ins("neg.f32", neg_mag, mag)
        signed = b.reg("f32")
        b.ins("selp.f32", signed, neg_mag, mag, odd)
        ei = b.reg("f32")
        b.ins("cvt.rzi.f32.f32", ei, expo)
        integral = b.reg("pred")
        b.ins("setp.eq.f32", integral, ei, expo)
        b.ins("selp.f32", signed, signed, f32_imm(float("nan")), integral)
        negative = b.reg("pred")
        b.ins("setp.lt.f32", negative, base, f32_imm(0.0))
        r = b.reg("f32")
        b.ins("selp.f32", r, signed, mag, negative)
        return r


def _int_cvt(dt: DType) -> str:
    """Integer type suffix for an integer (or bool) held in its register class."""
    signed = dt.kind == "int"
    bits = 64 if dt.reg == "s64" else 32
    return f"{'s' if signed else 'u'}{bits}"


def _mov_type(dt: DType) -> str:
    if not dt.is_float:
        return "b64" if dt.reg == "s64" else "b32"
    return {F16: "b16", F32: "f32", F64: "f64"}[dt]
