"""Lowering of the cross-thread combines: WarpShuffle and TreeHalve.

A mixin of ``KernelLowering`` (see lower.py). Both statements carry a small
combine program (a list of ``Assign``s) that merges two partial states; it is
lowered after every exchange, with assignments to state names updating the
state's register in place. Sum, max and online softmax all use this one path.
"""

from __future__ import annotations

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

from emmyptx import dtypes
from emmyptx.elementwise import Value
from emmyptx.memory import _ldst


class CombineLowering:
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
