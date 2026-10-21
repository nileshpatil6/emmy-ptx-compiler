"""Stage 2: whole-kernel analyses that run before any PTX is emitted.

Each answers one question about the statement tree that a statement-by-
statement walk could not answer on its own:

* ``read_only_buffers``: which inputs are never written by this kernel
  (their loads may use the read-only cache and be reused);
* ``carried_names``: which names are updated in place (accumulators);
* ``fusable_products``: which ``a * b`` products feed exactly one ``add``
  accumulation, so ``acc += a * b`` can become a single ``fma``;
* ``vector_load_groups``: which scalar f32 loads read neighbouring, aligned
  elements and can become one ``v2`` / ``v4`` vector load.
"""

from __future__ import annotations

from emmy.compiler.ir.elementwise import reduce_canon

from emmyptx import dtypes
from emmyptx.index import flatten, linearize
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


# Statements that neither write memory nor synchronize: loads may move across them.
_PURE = frozenset({"Load", "Assign", "Let", "Select", "Accum", "Init"})


def vector_load_groups(spec: KernelSpec, read_only: frozenset[str]) -> dict[int, tuple]:
    """Groups of scalar f32 loads that can be merged into one vector load.

    Inside one straight run of pure statements, loads of the same buffer whose
    linearized indices have identical variable parts and constants c, c+1, ...,
    c+w-1 read ``w`` neighbouring elements. The vector access is aligned when c
    and every coefficient are multiples of ``w`` (buffers themselves are 16-byte
    aligned). Global buffers must be read-only in this kernel; shared ones are
    safe because no write or barrier sits between the merged loads.

    Returns ``{id(first load): (load_0, ..., load_w-1)}``; the lowering emits the
    vector load at the first one and skips the others.
    """
    global_shapes = {n: b.shape for n, b in spec.buffers.items() if b.dtype is dtypes.F32 and n in read_only}
    smem_shapes = {s.name: tuple(s.extents) for s in walk(spec.body) if type(s).__name__ == "Smem" and dtypes.get(s.dtype) is dtypes.F32}
    bodies = [(spec.body, False)]
    bodies += [(child, type(s).__name__ == "Loop" and s.unroll) for s in walk(spec.body) for child in s.nested()]
    groups: dict[int, tuple] = {}
    for body, unrolled in bodies:
        # Shared loads only in loops we unroll: in a rolled loop ptxas pairs them
        # itself and measured faster without our help (m_rect: 5.2 vs 7.0 us).
        shapes = global_shapes | smem_shapes if unrolled else global_shapes
        run: list = []
        for s in [*body, None]:  # None flushes the last run
            if s is not None and type(s).__name__ in _PURE:
                run.append(s)
                continue
            groups |= _groups_in_run(run, shapes)
            run = []
    return groups


def _groups_in_run(run: list, shapes: dict) -> dict[int, tuple]:
    defined = {n for s in run for n in s.defines()}
    by_base: dict[tuple, dict[int, object]] = {}
    for s in run:
        if type(s).__name__ != "Load" or len(s.names) != 1 or s.input not in shapes:
            continue
        flat = flatten(shapes[s.input], s.index)
        if flat.free_vars() & defined:  # its index is computed inside the run
            continue
        terms, const = linearize(flat)
        by_base.setdefault((s.input, frozenset(terms.items())), {}).setdefault(const, s)
    groups = {}
    for (_, terms), loads in by_base.items():
        taken: set[int] = set()
        for c in sorted(loads):
            for w in (4, 2):
                lanes = [c + k for k in range(w)]
                if c % w or any(k % w for _, k in terms) or not all(x in loads and x not in taken for x in lanes):
                    continue
                members = [loads[x] for x in lanes]
                first = min(members, key=run.index)
                groups[id(first)] = tuple(members)
                taken.update(lanes)
                break
    return groups
