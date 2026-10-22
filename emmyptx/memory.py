"""Lowering of memory statements: Smem, Load, Write and the cp.async copies.

A mixin of ``KernelLowering`` (see lower.py): it reads the same environment
and builder. Its central method is ``_address``, which turns a buffer and a
flat element index into a ``[register+offset]`` PTX operand: the constant part
of the normalized index becomes the immediate offset, and the variable part's
register is computed once and reused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from emmy.compiler.ir.expr import BinaryExpr, Literal

from emmyptx import dtypes
from emmyptx.dtypes import F16, F32, DType
from emmyptx.elementwise import Value
from emmyptx.index import flatten, known_multiple, normalize, split_constant
from emmyptx.loader import CompileError
from emmyptx.ptx import SharedArray

STATIC_SMEM_LIMIT = 48 * 1024  # larger footprints must use dynamic shared memory


@dataclass(frozen=True)
class SharedBuffer:
    dtype: DType
    shape: tuple[int, ...]
    symbol: str  # PTX symbol the buffer lives in
    offset: int  # byte offset of the buffer within that symbol


class MemoryLowering:
    # ---- memory ------------------------------------------------------------
    def _stmt_Smem(self, s) -> None:
        if s.name in self.smem:
            return
        dt = dtypes.get(s.dtype)
        shape = tuple(int(e) for e in s.extents)
        nbytes = math.prod(shape) * dt.size
        align = max(16, int(s.align or 0), dt.size)
        if self.pool is not None:
            offset = -(-self.pool.nbytes // align) * align
            self.pool.nbytes = offset + nbytes
            self.smem[s.name] = SharedBuffer(dt, shape, self.pool.name, offset)
        else:
            symbol = f"smem_{s.name}"
            self.static_shared.append(SharedArray(symbol, nbytes, align))
            self.smem[s.name] = SharedBuffer(dt, shape, symbol, 0)

    def _check_shared_budget(self) -> None:
        if self.pool is not None:
            if self.pool.nbytes > self.spec.launch.dynamic_smem:
                raise CompileError(f"shared memory needs {self.pool.nbytes} bytes but the launch provides {self.spec.launch.dynamic_smem}")
        elif sum(a.nbytes for a in self.static_shared) > STATIC_SMEM_LIMIT:
            raise CompileError("static shared memory above 48 KiB needs a dynamic allocation in meta.json")

    def _buffer_dtype(self, name: str) -> DType:
        if name in self.smem:
            return self.smem[name].dtype
        if name in self.spec.buffers:
            return self.spec.buffers[name].dtype
        raise CompileError(f"unknown buffer {name!r}")

    def _flatten(self, name: str, index: tuple):
        """Row-major flat element offset of ``index`` in buffer ``name``."""
        return flatten(self.smem[name].shape if name in self.smem else self.spec.buffers[name].shape, index)

    def _address(self, name: str, flat, swizzle: str = "NONE") -> str:
        """``[reg+offset]`` addressing element ``flat`` of buffer ``name``.
        The constant part of the index becomes the immediate offset, so
        neighbouring accesses share one address register."""
        size = self._buffer_dtype(name).size
        flat = self.idx.specialize(flat)
        if swizzle != "NONE":
            flat = _swizzle(flat, swizzle)
        residual, const = split_constant(flat)
        if residual is not None and (c := self.idx.const(residual)) is not None:
            residual, const = None, const + c
        if name in self.smem:
            buf = self.smem[name]
            offset = buf.offset + const * size
            if residual is None:
                return _mem(buf.symbol, offset)

            def shared_addr() -> str:
                base = self.idx.cached(("symbol", buf.symbol), frozenset(), lambda: self._symbol_address(buf.symbol))
                a = self.b.reg("s32")
                self.b.ins("mad.lo.s32", a, self._reg_operand(self.idx.int_(residual)), str(size), base)
                return a

            reg = self.idx.cached(("saddr", name, residual), residual.free_vars(), shared_addr)
            return _mem(reg, offset)
        base = self.ptr[name]
        if residual is None:
            return _mem(base, const * size)

        def byte_offset() -> str:  # shared by every buffer indexed the same way
            wide = self.b.reg("s64")
            self.b.ins("mul.wide.s32", wide, self._reg_operand(self.idx.int_(residual)), str(size))
            return wide

        def global_addr() -> str:
            wide = self.idx.cached(("offset", residual, size), residual.free_vars(), byte_offset)
            a = self.b.reg("s64")
            self.b.ins("add.s64", a, base, wide)
            return a

        reg = self.idx.cached(("gaddr", name, residual), residual.free_vars(), global_addr)
        return _mem(reg, const * size)

    def _symbol_address(self, symbol: str) -> str:
        r = self.b.reg("s32")
        self.b.ins("mov.u32", r, symbol)
        return r

    def _reg_operand(self, operand: str) -> str:
        if operand.startswith("%"):
            return operand
        r = self.b.reg("s32")
        self.b.ins("mov.s32", r, operand)
        return r

    def _stmt_Load(self, s) -> None:
        if id(s) in self.merged_loads:  # already loaded by its group's vector load
            return
        if id(s) in self.load_groups:
            group = self.load_groups[id(s)]  # in lane order; s is the first in program order
            values = self._load_vector(s.input, self._flatten(s.input, group[0].index), F32, len(group))
            for load, v in zip(group, values, strict=True):
                self.bind(load.names[0], v)
            return
        dt = self._buffer_dtype(s.input)
        width = len(s.names)
        if width > 1 and _vector_ok(dt, width):
            values = self._load_vector(s.input, self._flatten(s.input, s.index), dt, width)
        else:
            values = [self._load_scalar(s.input, self._flatten(s.input, _lane_index(s.index, k)), dt) for k in range(width)]
        for name, v in zip(s.names, values, strict=True):
            self.bind(name, self.ew.convert(v, dtypes.compute_dtype(dt)))

    def _space(self, name: str) -> str:
        if name in self.smem:
            return "shared"
        return "global.nc" if name in self.read_only else "global"

    def _load_scalar(self, name: str, flat, dt: DType) -> Value:
        flat = self.idx.specialize(flat)

        def load() -> str:
            r = self.b.reg(dt.reg)
            self.b.ins(f"ld.{self._space(name)}.{_ldst(dt)}", r, self._address(name, flat))
            return r

        if name in self.read_only:  # an input never changes: load each element once
            return Value(self.idx.cached(("load", name, flat), flat.free_vars(), load), dt)
        return Value(load(), dt)

    def _load_vector(self, name: str, flat, dt: DType, width: int) -> list[Value]:
        b = self.b
        addr = self._address(name, flat)
        space = self._space(name)
        if dt is F32:
            regs = [b.reg("f32") for _ in range(width)]
            b.ins(f"ld.{space}.v{width}.f32", "{" + ", ".join(regs) + "}", addr)
            return [Value(r, dt) for r in regs]
        # f16: move pairs as 32-bit words, then split each word into two halves
        words = [b.reg("s32") for _ in range(width // 2)]
        if len(words) == 1:
            b.ins(f"ld.{space}.b32", words[0], addr)
        else:
            b.ins(f"ld.{space}.v{len(words)}.b32", "{" + ", ".join(words) + "}", addr)
        halves = []
        for w in words:
            lo, hi = b.reg("b16"), b.reg("b16")
            b.ins("mov.b32", f"{{{lo}, {hi}}}", w)
            halves += [Value(lo, F16), Value(hi, F16)]
        return halves

    def _stmt_Write(self, s) -> None:
        dt = self._buffer_dtype(s.output)
        values = [self.ew.convert(self.lookup(v), dt) for v in s.values]
        space = "shared" if s.output in self.smem else "global"
        if s.atomic:
            if space != "global" or len(values) != 1:
                raise CompileError("atomic Write must be a scalar global store")
            addr = self._address(s.output, self._flatten(s.output, s.index))
            self.b.ins(f"red.global.add.{_atomic_type(dt)}", addr, self._value_reg(values[0]))
            return
        width = len(values)
        if width > 1 and _vector_ok(dt, width):
            addr = self._address(s.output, self._flatten(s.output, s.index), s.swizzle)
            self._store_vector(space, addr, dt, values)
            return
        for k, v in enumerate(values):
            addr = self._address(s.output, self._flatten(s.output, _lane_index(s.index, k)), s.swizzle)
            self.b.ins(f"st.{space}.{_ldst(dt)}", addr, self._value_reg(v))

    def _store_vector(self, space: str, addr: str, dt: DType, values: list[Value]) -> None:
        b = self.b
        if dt is F32:
            b.ins(f"st.{space}.v{len(values)}.f32", addr, "{" + ", ".join(v.reg for v in values) + "}")
            return
        words = []
        for lo, hi in zip(values[::2], values[1::2], strict=True):
            w = b.reg("s32")
            b.ins("mov.b32", w, f"{{{lo.reg}, {hi.reg}}}")
            words.append(w)
        if len(words) == 1:
            b.ins(f"st.{space}.b32", addr, words[0])
        else:
            b.ins(f"st.{space}.v{len(words)}.b32", addr, "{" + ", ".join(words) + "}")

    def _value_reg(self, v: Value) -> str:
        return v.reg if v.reg.startswith("%") else self.ew.fresh(v).reg

    # ---- asynchronous copies into shared memory ----------------------------
    def _stmt_CpAsyncCopy(self, s) -> None:
        flat = self._flatten(s.smem, s.smem_index)
        if s.lane_index is not None:  # the lane's own (row, col) inside the stripe
            cols = self.smem[s.smem].shape[-1]
            row, col = s.lane_index
            flat = BinaryExpr("+", flat, BinaryExpr("+", BinaryExpr("*", row, Literal(cols, "int")), col))
        if s.nbytes not in (4, 8, 16):
            raise CompileError(f"cp.async of {s.nbytes} bytes is not supported")
        src_flat = self._flatten(s.src, s.src_index)
        dst = self._address(s.smem, flat, s.swizzle)
        src = self._address(s.src, src_flat)
        copy = ("cp.async.cg.shared.global", dst, src, "16") if s.nbytes == 16 else ("cp.async.ca.shared.global", dst, src, str(s.nbytes))
        if self._aligned(s.src, src_flat, s.nbytes) and (s.swizzle != "NONE" or self._aligned(s.smem, flat, s.nbytes)):
            self.b.ins(*copy)
            return
        # cp.async needs both addresses aligned to its size. When that cannot be
        # proved, test it at run time and copy element by element otherwise
        # (the IR's wait + barrier before the data is read cover both paths).
        b = self.b
        g, sh = self._address_register(src, "s64"), self._address_register(dst, "s32")
        mg, ms = b.reg("s64"), b.reg("s32")
        b.ins("and.b64", mg, g, str(s.nbytes - 1))
        b.ins("and.b32", ms, sh, str(s.nbytes - 1))
        b.ins("or.b32", ms, ms, self.ew.convert(Value(mg, dtypes.U64), dtypes.U32).reg)
        p = b.reg("pred")
        b.ins("setp.ne.s32", p, ms, "0")
        slow, done = b.label("unaligned_copy"), b.label("copy_done")
        b.branch(slow, pred=p)
        b.ins(*copy)
        b.branch(done)
        b.place(slow)
        dt = self._buffer_dtype(s.src)
        for k in range(s.nbytes // dt.size):
            r = b.reg(dt.reg)
            b.ins(f"ld.global.{_ldst(dt)}", r, f"[{g}+{k * dt.size}]")
            b.ins(f"st.shared.{_ldst(dt)}", f"[{sh}+{k * dt.size}]", r)
        b.place(done)

    def _aligned(self, name: str, flat, nbytes: int) -> bool:
        """Is element ``flat`` of ``name`` provably ``nbytes``-aligned? Buffers
        start 16-byte aligned, so it is enough that the byte offset is a multiple."""
        offset = self.smem[name].offset if name in self.smem else 0
        multiple = known_multiple(normalize(self.idx.specialize(flat)), self.var_multiples) * self._buffer_dtype(name).size
        return offset % nbytes == 0 and (multiple % nbytes == 0)

    def _address_register(self, addr: str, cls: str) -> str:
        """The full address of a ``[base+offset]`` operand in one register."""
        base, _, offset = addr.strip("[]").partition("+")
        if not base.startswith("%"):  # a shared-memory symbol
            base = self._symbol_address(base)
        if not offset:
            return base
        r = self.b.reg(cls)
        self.b.ins(f"add.{'s64' if cls == 's64' else 's32'}", r, base, offset)
        return r

    def _stmt_CpAsyncCommit(self, s) -> None:
        self.b.ins("cp.async.commit_group")

    def _stmt_CpAsyncWait(self, s) -> None:
        self.b.ins("cp.async.wait_group", str(s.group))


def _mem(base: str, offset: int) -> str:
    return f"[{base}]" if offset == 0 else f"[{base}+{offset}]"


def _lane_index(index: tuple, k: int) -> tuple:
    """Index of element ``k`` of a grouped access: the last coordinate + k."""
    if k == 0:
        return index
    if not index:
        return (Literal(k, "int"),)
    return (*index[:-1], BinaryExpr("+", index[-1], Literal(k, "int")))


def _vector_ok(dt: DType, width: int) -> bool:
    """Widths emmy emits as one vector access (it guarantees the alignment)."""
    return (dt is F32 and width in (2, 4)) or (dt is F16 and width in (2, 4, 8))


def _ldst(dt: DType) -> str:
    return "b16" if dt is F16 else dt.mem


def _atomic_type(dt: DType) -> str:
    return {"f32": "f32", "f64": "f64", "f16": "noftz.f16", "i32": "s32", "u32": "u32", "i64": "u64", "u64": "u64"}[dt.name]


def _swizzle(flat, mode: str):
    """emmy's XOR slab swizzle ``e ^ (((e >> shift) & mask) << 3)``."""
    table = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
    base, _, shift_override = mode.partition("@")
    if base not in table:
        raise CompileError(f"smem swizzle mode {mode!r} is not supported")
    shift, mask = table[base]
    if shift_override:
        shift = int(shift_override)
    lit = lambda v: Literal(v, "int")  # noqa: E731
    bits = BinaryExpr("<<", BinaryExpr("&", BinaryExpr(">>", flat, lit(shift)), lit(mask)), lit(3))
    return BinaryExpr("^", flat, bits)
