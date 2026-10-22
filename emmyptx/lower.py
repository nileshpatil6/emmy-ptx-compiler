"""Stage 3: walk the Kernel IR statement tree and emit PTX for each statement.

``KernelLowering`` keeps one environment that maps every IR name (axis
variables, SSA values, accumulators) to a ``Value`` (register + dtype) and
dispatches each statement to a ``_stmt_<Kind>`` method. Control flow is
structured: loops and conditions become labels and predicated branches.

Memory statements are in memory.py and the cross-thread combines in
combine.py (mixins of the same class). Statement kinds and their PTX:

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

from emmy.compiler.ir.elementwise import reduce_canon
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from emmyptx import dtypes
from emmyptx.analysis import carried_names, fusable_products, read_only_buffers, statement_count, vector_load_groups
from emmyptx.combine import CombineLowering
from emmyptx.dtypes import F32, I32
from emmyptx.elementwise import Elementwise, Value
from emmyptx.index import IndexLowerer, is_imm
from emmyptx.loader import CompileError, KernelSpec
from emmyptx.memory import MemoryLowering, SharedBuffer
from emmyptx.ptx import PtxBuilder, SharedArray, render_kernel

GID = "__gid"  # internal name of the flat cell id (not an IR name)
UNROLL_BUDGET = 4096  # statements a fully unrolled loop may expand to
# Other innermost loops with a constant trip count run their body 4 times per
# trip. Measured on the runner against no unrolling and full unrolling: 10%
# faster on average over the kernels above 15 us, up to 3x on attention,
# never more than noise slower (RESULTS.md).
PARTIAL_UNROLL = 4


def lower_kernel(spec: KernelSpec) -> str:
    return KernelLowering(spec).run()


class KernelLowering(MemoryLowering, CombineLowering):
    """Lowers one kernel; memory and combine statements live in the mixins."""

    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.b = PtxBuilder()
        self.ew = Elementwise(self.b)
        self.idx = IndexLowerer(self.b, self._index_operand)
        self.env: dict[str, Value] = {}
        self.var_multiples: dict[str, int] = {}  # loop counter -> a number it is always a multiple of
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
        self.load_groups = vector_load_groups(spec, self.read_only)
        self.merged_loads = {id(m) for g in self.load_groups.values() for m in g[1:]}

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
        self.var_multiples.pop(name, None)
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
            self.ew.loop_depth += 1
            for k in range(trip):  # one straight-line block: copies share values
                self.bind(s.axis.name, Value(str(k), I32))
                self._body(s.body)
            self.ew.loop_depth -= 1
            self.idx.pop_scope()
            return
        self.b.comment(f"for {s.axis.name} in 0..{trip}")
        innermost = not any(type(c).__name__ in ("Loop", "StridedLoop") for c in s.body)
        unroll = PARTIAL_UNROLL if innermost and trip % PARTIAL_UNROLL == 0 and trip > PARTIAL_UNROLL else 1
        self._counted_loop(s.axis.name, "0", str(trip), "1", s.body, unroll)

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

    def _counted_loop(self, var: str, start: str, end: str, step: str, body, unroll: int = 1) -> None:
        """``for var = start; var < end; var += step``. With ``unroll`` > 1 (the
        trip count must be a multiple) each trip runs the body ``unroll`` times,
        with the counter rebound to ``var + k * step`` for copy ``k``."""
        b = self.b
        i = b.reg("s32")
        b.ins("mov.s32", i, start)
        self.bind(var, Value(i, I32))
        if is_imm(start) and is_imm(step):  # start, start + step, ... share their common divisor
            self.var_multiples[var] = math.gcd(int(start), int(step))
        top, done = b.label(f"{var}_top"), b.label(f"{var}_done")
        b.place(top)
        p = b.reg("pred")
        b.ins("setp.ge.s32", p, i, end)
        b.branch(done, pred=p)
        self.ew.loop_depth += 1
        if unroll == 1:
            self._block(body)
        else:
            self.idx.push_scope()  # one straight-line block: copies share values
            for k in range(unroll):
                if k:
                    ik = b.reg("s32")
                    b.ins("add.s32", ik, i, str(k * int(step)))
                    self.bind(var, Value(ik, I32))
                self._body(body)
            self.idx.pop_scope()
            self.bind(var, Value(i, I32))
            step = str(unroll * int(step))
        self.ew.loop_depth -= 1
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
