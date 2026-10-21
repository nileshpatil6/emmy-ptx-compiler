"""Stage 2: whole-kernel analyses that run before any PTX is emitted.

Each answers one question about the statement tree that a statement-by-
statement walk could not answer on its own:

* ``read_only_buffers``: which inputs are never written by this kernel
  (their loads may use the read-only cache and be reused);
* ``carried_names``: which names are updated in place (accumulators);
* ``fusable_products``: which ``a * b`` products feed exactly one ``add``
  accumulation, so ``acc += a * b`` can become a single ``fma``.
"""

from __future__ import annotations

from emmy.compiler.ir.elementwise import reduce_canon

from emmyptx.loader import KernelSpec


def walk(stmts):
    for s in stmts:
        yield s
        for child in s.nested():
            yield from walk(child)


def statement_count(stmts) -> int:
    return sum(1 for _ in walk(stmts))


def read_only_buffers(spec: KernelSpec) -> frozenset[str]:
    written = {s.output for s in walk(spec.body) if type(s).__name__ == "Write"}
    written |= {s.dst for s in walk(spec.body) if type(s).__name__ == "ZeroPrologue"}
    return frozenset(n for n in spec.params if n not in written and n not in spec.outputs)


def carried_names(body) -> frozenset[str]:
    """Names whose register is updated in place (accumulators, Init state)."""
    return frozenset(s.name for s in walk(body) if type(s).__name__ in ("Accum", "Init"))


def fusable_products(body, carried: frozenset[str]) -> frozenset[str]:
    """f32 products read exactly once, as the value of an ``add`` Accum in the
    same block: ``acc += a * b`` then becomes one ``fma`` instruction."""
    uses: dict[str, int] = {}
    for s in walk(body):
        for name in _reads(s):
            uses[name] = uses.get(name, 0) + 1
    fusable = set()
    for s in walk(body):
        for child in s.nested():
            products = {}
            for c in child:
                kind = type(c).__name__
                if kind == "Assign" and c.op.name == "multiply" and len(c.args) == 2 and not (set(c.args) & carried):
                    if c.dtype is None or c.dtype.name == "f32":
                        products[c.name] = c
                elif kind == "Accum" and c.value in products and uses.get(c.value) == 1:
                    if reduce_canon(c.op.name) == "add" and (c.dtype is None or c.dtype.name == "f32"):
                        fusable.add(c.value)
    return frozenset(fusable)


def _reads(s) -> tuple[str, ...]:
    kind = type(s).__name__
    if kind == "Assign":
        return tuple(s.args)
    if kind == "Accum":
        return (s.value,) + ((s.base,) if s.base else ())
    if kind == "Write":
        return tuple(s.values)
    if kind == "Select":
        return tuple(b.value for b in s.branches)
    if kind in ("WarpShuffle", "TreeHalve"):
        return tuple(x for a in s.combine_states for x in a.args)
    return ()
