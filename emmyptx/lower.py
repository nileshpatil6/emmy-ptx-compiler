"""Stage 3: walk the Kernel IR statement tree and emit PTX for each statement.

``KernelLowering`` keeps one environment that maps every IR name (axis
variables, SSA values, accumulators) to a ``Value`` (register + dtype) and
dispatches each statement to a ``_stmt_<Kind>`` method. Control flow is
structured: loops and conditions become labels and predicated branches.

Statement kinds and their PTX, at a glance:

    Tile            thread id -> global cell id -> per-axis indices (+ guard)
    Load / Write    ld / st (global or shared), v2 / v4 when the IR groups them
    Assign / Let    arithmetic (see elementwise.py) / a literal or an integer
    Accum / Init    in-place update of a register that lives across a loop
    Loop            counter register, compare, branch back
    StridedLoop     same, starting at the thread's lane and stepping by the
                    number of cooperating threads
    Cond            setp + predicated branch around the body
    Smem / Sync     .shared array / bar.sync
    WarpShuffle     shfl.sync.bfly butterfly over the warp's registers
    TreeHalve       halving tree over a shared-memory array, then broadcast
    CpAsync*        cp.async global -> shared copies, commit, wait
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from emmy.compiler.ir.elementwise import reduce_canon
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from emmyptx import dtypes
from emmyptx.analysis import carried_names, fusable_products, read_only_buffers, statement_count
from emmyptx.dtypes import F16, F32, I32, DType
from emmyptx.elementwise import Elementwise, Value
from emmyptx.index import IndexLowerer, split_constant
from emmyptx.loader import CompileError, KernelSpec
from emmyptx.ptx import PtxBuilder, SharedArray, render_kernel

STATIC_SMEM_LIMIT = 48 * 1024  # larger footprints must use dynamic shared memory
GID = "__gid"  # internal name of the flat cell id (not an IR name)
UNROLL_BUDGET = 4096  # statements a fully unrolled loop may expand to


@dataclass(frozen=True)
class SharedBuffer:
    dtype: DType
    shape: tuple[int, ...]
    symbol: str  # PTX symbol the buffer lives in
    offset: int  # byte offset of the buffer within that symbol


def lower_kernel(spec: KernelSpec) -> str:
    return KernelLowering(spec).run()


class KernelLowering:
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.b = PtxBuilder()
        self.ew = Elementwise(self.b)
        self.idx = IndexLowerer(self.b, self._index_operand)
        self.env: dict[str, Value] = {}
        self.ptr: dict[str, str] = {}  # global buffer -> register holding its address
        self.smem: dict[str, SharedBuffer] = {}
        self.static_shared: list[SharedArray] = []
        launch = spec.launch
        self.pool = SharedArray("smem_pool", 0, 16) if launch and launch.dynamic_smem > 0 else None
        self.exit = self.b.label("exit")
        self.read_only = read_only_buffers(spec)
        self.carried = carried_names(spec.body)
        self.fused_products = fusable_products(spec.body, self.carried)
        self.deferred: dict[str, tuple[Value, Value]] = {}

    # ---- driver ------------------------------------------------------------
    def run(self) -> str:
        self._prologue()
        self._body(self.spec.body)
        self.b.place(self.exit)
        self.b.ins("ret")
        self._check_shared_budget()
        launch = self.spec.launch
        header = [f"Kernel IR -> PTX (emmyptx): {self.spec.name}"]
        if launch:
            header.append(f"launch: grid {list(launch.grid)} block {list(launch.block)} dynamic smem {launch.dynamic_smem}")
        return render_kernel(
            name=self.spec.name,
            params=[f"param_{p}" for p in self.spec.params],
            body=self.b,
            shared=self.static_shared,
            dynamic_shared=self.pool,
            max_threads=launch.block[0] * launch.block[1] * launch.block[2] if launch else None,
            header_comment=header,
        )

    def _prologue(self) -> None:
        b = self.b
        b.comment("buffer parameters -> global addresses")
        for name in self.spec.params:
            raw, addr = b.reg("s64"), b.reg("s64")
            b.ins("ld.param.u64", raw, f"[param_{name}]")
            b.ins("cvta.to.global.u64", addr, raw)
            self.ptr[name] = addr
        b.comment("thread coordinates")
        self.tid, self.ctaid, self.ntid = b.reg("s32"), b.reg("s32"), b.reg("s32")
        b.ins("mov.u32", self.tid, "%tid.x")
        b.ins("mov.u32", self.ctaid, "%ctaid.x")
        b.ins("mov.u32", self.ntid, "%ntid.x")
        lane, warp = b.reg("s32"), b.reg("s32")
        b.ins("and.b32", lane, self.tid, "31")
        b.ins("shr.u32", warp, self.tid, "5")
        self.bind("lane", Value(lane, I32))
        self.bind("warp", Value(warp, I32))

    def _body(self, stmts) -> None:
        for s in stmts:
            handler = getattr(self, f"_stmt_{type(s).__name__}", None)
            if handler is None:
                raise CompileError(f"unsupported Kernel IR statement: {type(s).__name__}")
            handler(s)

    def _block(self, stmts) -> None:
        """A nested block: values computed inside it stay inside it."""
        self.idx.push_scope()
        self._body(stmts)
        self.idx.pop_scope()

    # ---- environment -------------------------------------------------------
    def bind(self, name: str, value: Value) -> None:
        self.env[name] = value
        self.idx.invalidate(name)

    def lookup(self, name: str) -> Value:
        if name in self.deferred:  # a product fused into its accumulation
            raise CompileError(f"internal: fused product {name!r} read outside its Accum")
        try:
            return self.env[name]
        except KeyError:
            raise CompileError(f"name {name!r} is used before it is defined") from None

    def _index_operand(self, name: str) -> str:
        v = self.lookup(name)
        if not v.dtype.is_int:
            raise CompileError(f"{name!r} is used as an index but holds {v.dtype.name}")
        return self.ew.convert(v, I32).reg if v.dtype.reg == "s64" else v.reg

    def _assign_in_place(self, name: str, value: Value) -> None:
        self.ew.move(self.lookup(name), value)
        self.idx.invalidate(name)

    def _update_in_place(self, name: str, op: str, args: list[Value]) -> None:
        """``name = op(args)`` written into ``name``'s own register (an
        accumulator or a combine state keeps one register for its lifetime)."""
        target = self.lookup(name)
        if target.dtype is F32:
            self.ew.apply(op, args, into=target)
            self.idx.invalidate(name)
        else:
            self._assign_in_place(name, self.ew.apply(op, args, target.dtype))

    # ---- launch geometry ---------------------------------------------------
    def _stmt_Tile(self, t) -> None:
        b = self.b
        extents = [int(a.extent.as_static()) for a in t.axes]
        n = math.prod(extents)
        if n >= 2**31:
            raise CompileError(f"Tile with {n} cells needs 64-bit indexing, which is not supported")
        b.comment(f"Tile[{', '.join(a.name for a in t.axes)}] (N={n})")
        gid = b.reg("s32")
        if t.aux_threads:
            # producer threads past the cooperative cells wrap onto the CTA's first cells
            local = b.reg("s32")
            b.ins("rem.u32", local, self.tid, str(t.block_threads))
            b.ins("mad.lo.s32", gid, self.ctaid, str(t.block_threads), local)
        else:
            b.ins("mad.lo.s32", gid, self.ctaid, self.ntid, self.tid)
        launch = self.spec.launch
        if t.aux_threads or launch is None or launch.threads != n:
            p = b.reg("pred")
            b.ins("setp.ge.s32", p, gid, str(n))
            b.branch(self.exit, pred=p)  # threads past the last cell do nothing
        self.bind(GID, Value(gid, I32))
        for axis, expr in zip(t.axes, _decode_axes(t, extents), strict=True):
            self.bind(axis.name, Value(self.idx.int_(expr), I32))
        self._body(t.body)

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
        shape = self.smem[name].shape if name in self.smem else self.spec.buffers[name].shape
        if not index:
            return Literal(0, "int")
        if len(index) == 1:
            return index[0]
        if len(shape) != len(index):  # emmy's convention for a rank mismatch
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

    # ---- scalar compute ----------------------------------------------------
    def _stmt_Assign(self, s) -> None:
        if s.name in self.fused_products:
            # read by exactly one Accum: emitted there as a single fma
            self.deferred[s.name] = tuple(self.lookup(a) for a in s.args)
            return
        result = dtypes.compute_dtype(dtypes.get(s.dtype)) if s.dtype is not None else None
        self.bind(s.name, self.ew.apply(s.op.name, [self.lookup(a) for a in s.args], result))

    def _stmt_Let(self, s) -> None:
        dt = dtypes.compute_dtype(dtypes.get(s.dtype)) if s.dtype is not None else None
        if isinstance(s.value, Literal) and (isinstance(s.value.value, float) or s.value.dtype == "float"):
            self.bind(s.name, self.ew.constant(float(s.value.value), dt or F32))
            return
        v = Value(self.idx.int_(s.value), I32)
        self.bind(s.name, self.ew.convert(v, dt) if dt is not None and dt is not I32 else v)

    def _stmt_Select(self, s) -> None:
        """``name = value_i`` for the first branch whose predicate holds (the
        last branch is the catch-all), as a chain of ``selp``."""
        *conditional, last = s.branches
        result = self.ew.convert(self.lookup(last.value), F32)
        for br in reversed(conditional):
            value = self.ew.convert(self.lookup(br.value), F32)
            known = self.idx.const(br.select)
            if known is not None:
                result = value if known else result
                continue
            r = self.b.reg("f32")
            self.b.ins("selp.f32", r, value.reg, result.reg, self.idx.pred(br.select))
            result = Value(r, F32)
        self.bind(s.name, result)

    def _stmt_Init(self, s) -> None:
        dt = dtypes.compute_dtype(dtypes.get(s.dtype))
        self.bind(s.name, self.ew.fresh(self.ew.constant(s.identity, dt)))

    def _stmt_Accum(self, s) -> None:
        acc = self.lookup(s.name)
        base = acc if s.base is None or s.base == s.name else self.lookup(s.base)
        op = reduce_canon(s.op.name)
        product = self.deferred.pop(s.value, None)
        if product is not None:  # acc = base + a * b as one fused multiply-add
            a, c = (self.ew.convert(v, F32).reg for v in product)
            self.b.ins("fma.rn.f32", acc.reg, a, c, self.ew.convert(base, F32).reg)
            self.idx.invalidate(s.name)
            return
        value = self.lookup(s.value)
        if op not in ("add", "multiply", "maximum", "minimum"):
            raise CompileError(f"Accum combine {s.op.name!r} is not supported")
        self._update_in_place(s.name, op, [base, value])

    def _seed_accumulators(self, loop) -> None:
        """Every accumulator folded directly in ``loop`` starts at its op's
        identity just before the loop (``seed=False``: an Init declared it)."""
        seen = set()
        for s in loop.body:
            if type(s).__name__ != "Accum" or s.name in seen:
                continue
            seen.add(s.name)
            if not loop.seed:
                self.lookup(s.name)  # must already exist
                continue
            identity = s.op.identity
            if identity is None:
                raise CompileError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            dt = dtypes.compute_dtype(dtypes.get(s.dtype))
            self.bind(s.name, self.ew.fresh(self.ew.constant(identity, dt)))

    # ---- control flow ------------------------------------------------------
    def _stmt_Loop(self, s) -> None:
        self._seed_accumulators(s)
        trip = int(s.axis.extent.as_static())
        if s.unroll and trip * statement_count(s.body) <= UNROLL_BUDGET:
            # Fully unrolled: each copy sees the counter as a constant, so its
            # index arithmetic folds into immediate address offsets.
            self.b.comment(f"for {s.axis.name} in 0..{trip} (unrolled)")
            self.idx.push_scope()
            for k in range(trip):  # one straight-line block: copies share values
                self.bind(s.axis.name, Value(str(k), I32))
                self._body(s.body)
            self.idx.pop_scope()
            return
        self.b.comment(f"for {s.axis.name} in 0..{trip}")
        self._counted_loop(s.axis.name, "0", str(trip), "1", s.body)

    def _stmt_StridedLoop(self, s) -> None:
        self._seed_accumulators(s)
        var = s.axis.name
        start = self.idx.int_(s.start)
        step = self.idx.int_(s.step) if not isinstance(s.step, int) else str(s.step)
        if s.end is not None:
            end = self.idx.int_(s.end)
            self.bind(f"{var}_end", Value(end, I32))  # the body may read the bound
        else:
            end = str(int(s.axis.extent.as_static()))
        self.b.comment(f"for {var} in {s.start.pretty()}..{end}:{step}")
        self._counted_loop(var, start, end, step, s.body)

    def _counted_loop(self, var: str, start: str, end: str, step: str, body) -> None:
        b = self.b
        i = b.reg("s32")
        b.ins("mov.s32", i, start)
        self.bind(var, Value(i, I32))
        top, done = b.label(f"{var}_top"), b.label(f"{var}_done")
        b.place(top)
        p = b.reg("pred")
        b.ins("setp.ge.s32", p, i, end)
        b.branch(done, pred=p)
        self._block(body)
        b.ins("add.s32", i, i, step)
        self.idx.invalidate(var)
        b.branch(top)
        b.place(done)

    def _stmt_Cond(self, s) -> None:
        known = self.idx.const(s.cond)
        if known is not None:  # a compile-time constant condition
            self._block(s.body if known else s.else_body)
            return
        b = self.b
        b.comment(f"if {s.cond.pretty()}")
        p = self.idx.pred(s.cond)
        skip = b.label("else" if s.else_body else "endif")
        b.branch(skip, pred=f"!{p}")
        self._block(s.body)
        if s.else_body:
            end = b.label("endif")
            b.branch(end)
            b.place(skip)
            self._block(s.else_body)
            b.place(end)
        else:
            b.place(skip)

    def _stmt_Sync(self, s) -> None:
        if s.warp:
            self.b.ins("bar.warp.sync", "0xffffffff")
        else:
            self._barrier(s.barrier_id, s.count)

    # ---- cross-thread combines ---------------------------------------------
    def _merge(self, program, state: tuple[str, ...]) -> None:
        """Run a combine program (``Assign``s); assignments to a state name
        update that state's register in place."""
        for a in program:
            args = [self.lookup(x) for x in a.args]
            if a.name in state:
                self._update_in_place(a.name, a.op.name, args)
            else:
                self.bind(a.name, self.ew.apply(a.op.name, args))

    def _stmt_WarpShuffle(self, s) -> None:
        b = self.b
        b.comment(f"WarpShuffle({', '.join(s.state)}, length={s.length})")
        mask = b.reg("s32")
        b.ins("activemask.b32", mask)
        step = s.length // 2
        while step > 0:
            for st, sb in zip(s.state, s.state_b, strict=True):
                self.bind(sb, self._shuffle_xor(self.lookup(st), step, mask))
            self._merge(s.combine_states, s.state)
            step //= 2

    def _shuffle_xor(self, v: Value, lane_mask: int, mask: str) -> Value:
        """The value ``v`` held by lane ``laneid ^ lane_mask``."""
        b = self.b

        def shfl(word: str) -> str:
            out = b.reg("s32")
            b.ins("shfl.sync.bfly.b32", out, word, str(lane_mask), "31", mask)
            return out

        if v.dtype.reg in ("f32", "s32"):
            out = b.reg(v.dtype.reg)
            b.ins("shfl.sync.bfly.b32", out, self._value_reg(v), str(lane_mask), "31", mask)
            return Value(out, v.dtype)
        if v.dtype.reg in ("f64", "s64"):  # two 32-bit halves
            lo, hi = b.reg("s32"), b.reg("s32")
            b.ins("mov.b64", f"{{{lo}, {hi}}}", v.reg)
            out = b.reg(v.dtype.reg)
            b.ins("mov.b64", out, f"{{{shfl(lo)}, {shfl(hi)}}}")
            return Value(out, v.dtype)
        word = b.reg("s32")  # f16: widen to 32 bits and back
        b.ins("cvt.u32.u16", word, v.reg)
        out = b.reg("b16")
        b.ins("cvt.u16.u32", out, shfl(word))
        return Value(out, v.dtype)

    def _stmt_TreeHalve(self, s) -> None:
        """Halving tree: while s > 0, threads t < s fold slot t+s into slot t;
        then every thread reads the result from the root slot."""
        b = self.b
        b.comment(f"TreeHalve({', '.join(s.state)}, length={s.length}, tid={s.tid_var})")
        t = Var(s.tid_var)
        if s.inner is not None:
            inner_var, scale = s.inner

            def slot(k):
                return BinaryExpr("+", BinaryExpr("*", BinaryExpr("+", t, Literal(k, "int")), Literal(scale, "int")), Var(inner_var))

            root = Var(inner_var)
        else:

            def slot(k):
                return BinaryExpr("+", t, Literal(k, "int"))

            root = Literal(0, "int")
        outer = {n: self.env.get(n) for n in (*s.state, *s.state_b)}
        dt = dtypes.compute_dtype(dtypes.get(s.dtype))
        half = s.length // 2
        while half > 0:
            skip = b.label("tree_skip")
            p = self.idx.pred(BinaryExpr(">=", t, Literal(half, "int")))
            b.branch(skip, pred=p)
            self.idx.push_scope()
            for buf, st, sb in zip(s.bufs, s.state, s.state_b, strict=True):  # shadow copies of the state
                self.bind(st, self.ew.convert(self._load_scalar(buf, slot(0), self._buffer_dtype(buf)), dt))
                self.bind(sb, self.ew.convert(self._load_scalar(buf, slot(half), self._buffer_dtype(buf)), dt))
            self._merge(s.combine_states, s.state)
            for buf, st in zip(s.bufs, s.state, strict=True):
                v = self.ew.convert(self.lookup(st), self._buffer_dtype(buf))
                self.b.ins(f"st.shared.{_ldst(v.dtype)}", self._address(buf, slot(0)), v.reg)
            self.idx.pop_scope()
            b.place(skip)
            self._barrier(s.barrier_id, s.barrier_count)
            half //= 2
        for name, v in outer.items():  # leave the shadows; back to the carried registers
            if v is not None:
                self.bind(name, v)
        for buf, st in zip(s.bufs, s.state, strict=True):
            self._assign_in_place(st, self._load_scalar(buf, root, self._buffer_dtype(buf)))

    def _barrier(self, barrier_id: int, count: int | None) -> None:
        if barrier_id == 0:
            self.b.ins("bar.sync", "0")
        else:
            self.b.ins("bar.sync", str(barrier_id), str(count))

    # ---- asynchronous copies into shared memory ----------------------------
    def _stmt_CpAsyncCopy(self, s) -> None:
        flat = self._flatten(s.smem, s.smem_index)
        if s.lane_index is not None:  # the lane's own (row, col) inside the stripe
            cols = self.smem[s.smem].shape[-1]
            row, col = s.lane_index
            flat = BinaryExpr("+", flat, BinaryExpr("+", BinaryExpr("*", row, Literal(cols, "int")), col))
        dst = self._address(s.smem, flat, s.swizzle)
        src = self._address(s.src, self._flatten(s.src, s.src_index))
        if s.nbytes == 16:
            self.b.ins("cp.async.cg.shared.global", dst, src, "16")
        elif s.nbytes in (4, 8):
            self.b.ins("cp.async.ca.shared.global", dst, src, str(s.nbytes))
        else:
            raise CompileError(f"cp.async of {s.nbytes} bytes is not supported")

    def _stmt_CpAsyncCommit(self, s) -> None:
        self.b.ins("cp.async.commit_group")

    def _stmt_CpAsyncWait(self, s) -> None:
        self.b.ins("cp.async.wait_group", str(s.group))


# ---- helpers ---------------------------------------------------------------
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


def _decode_axes(t, extents: list[int]) -> list:
    """Index expression of each Tile axis in terms of the flat cell id:
    ``a_i = (gid / stride_i) % extent_i`` (row-major, last axis fastest),
    optionally with emmy's grouped CTA rasterization of two block axes."""
    gid = Var(GID)
    lit = lambda v: Literal(v, "int")  # noqa: E731
    strides = [math.prod(extents[i + 1 :]) for i in range(len(extents))]
    exprs = []
    for i in range(len(extents)):
        term = gid if strides[i] == 1 else BinaryExpr("/", gid, lit(strides[i]))
        exprs.append(term if i == 0 else BinaryExpr("%", term, lit(extents[i])))
    names = [a.name for a in t.axes]
    if not (t.raster_group and t.raster_axes and all(n in names for n in t.raster_axes)):
        return exprs
    mi, ni = names.index(t.raster_axes[0]), names.index(t.raster_axes[1])
    if ni != mi + 1:
        return exprs
    gi, oi = (mi, ni) if t.raster_orient == "m" else (ni, mi)
    g = min(t.raster_group, extents[gi])
    if extents[gi] <= 1 or g <= 1:
        return exprs
    ea, eb = extents[gi], extents[oi]
    sub = gid if strides[ni] == 1 else BinaryExpr("/", gid, lit(strides[ni]))
    if mi != 0:
        sub = BinaryExpr("%", sub, lit(ea * eb))
    within = BinaryExpr("%", sub, lit(g * eb))
    if ea % g == 0:
        a_expr = BinaryExpr("+", BinaryExpr("*", BinaryExpr("/", sub, lit(g * eb)), lit(g)), BinaryExpr("%", sub, lit(g)))
        b_expr = BinaryExpr("/", within, lit(g))
    else:
        grp = BinaryExpr("/", sub, lit(g * eb))
        rest = BinaryExpr("-", lit(ea), BinaryExpr("*", grp, lit(g)))
        size = TernaryExpr(BinaryExpr("<", lit(g), rest), lit(g), rest)
        a_expr = BinaryExpr("+", BinaryExpr("*", grp, lit(g)), BinaryExpr("%", within, size))
        b_expr = BinaryExpr("/", within, size)
    exprs[gi], exprs[oi] = a_expr, b_expr
    return exprs
