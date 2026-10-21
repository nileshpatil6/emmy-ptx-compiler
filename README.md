# emmyptx: Emmy Kernel IR to PTX

A compiler for the [Emmy Hackathon 2026](https://riftstack.ai/hackathon). It reads one
kernel of Emmy's Kernel IR (json) and prints NVIDIA PTX that `ptxas` assembles directly:

```bash
pip install emmy-ml==0.3.11 numpy
python compile.py path/to/kernel.ir.json > out.ptx
```

The IR to PTX lowering is entirely our own Python (about 1,700 lines in `emmyptx/`). Emmy is
used only to deserialize the IR (`Graph.from_dict`); no emmy renderer, nvcc, NVRTC, LLVM or
other compiler is involved. `ptxas` only ever runs on the hackathon's remote runner.

## Results

All 15 open problems and all 82 generated variants pass on the runner's RTX 5090, across all
five categories. Full table with timings: [RESULTS.md](RESULTS.md).

| category | open set | generated variants |
|---|---|---|
| 1 pointwise | 4/4 | 31/31 |
| 2 reductions | 4/4 | 20/20 |
| 3 matmul | 2/2 | 12/12 |
| 4 fused | 3/3 | 12/12 |
| 5 attention-like | 2/2 | 7/7 |

The generated variants are our stand-in for the hidden set: 82 torch expressions with new
shapes (odd sizes, sizes that do not divide the block, tiny and large), new ops (exp, log,
tanh, sigmoid, silu, gelu, erf, sin, cos, sqrt, rsqrt, pow, where, clamp, max/min, abs),
broadcasting, f16 / f64 / int64 data, other reduction axes and block sizes, schedules that
split a reduction across blocks with atomics, matmuls with odd K, MLPs, layernorm, softmax,
variance, and attention of several shapes. They are traced and packaged by the organizers'
own `tools/generate.py`, exactly like the hidden set.

## How to run

```bash
python compile.py tests/open/r01_array_sum/k_sum_1_reduce.ir.json     # print PTX
python -m pytest tests                                                # offline checks, no GPU
python tools/run_all.py                                               # everything on the runner
python tools/run_all.py -k m0 --timing                                # a subset, with timings
python tools/run_all.py --markdown RESULTS_table.md                   # regenerate the table
```

`tools/run_all.py` does what the hackathon's `submit.py --compiler compile.py` does for every
problem folder (compile, assemble, launch with `meta.json`'s geometry, chain intermediate
buffers, compare with `expected.npz`), caches runner answers in `.cache/` so unchanged kernels
are never resent, and backs off when the runner is busy. In a sandbox that only allows HTTPS
CONNECT, set `RUNNER_PROXY=host:port`.

To regenerate the variants (needs torch as well):
`python tools/gen_variants.py --hackathon ../emmy-hackathon-2026`.

## Architecture

```
  kernel.ir.json + meta.json
            |
            v
  loader.py      load the IR with Graph.from_dict, buffers (shape, dtype),
                 parameter order and launch geometry from meta.json
            |
            v
  analysis.py    whole-kernel facts: read-only inputs, accumulators,
                 products that can fuse into an fma
            |
            v
  lower.py       walk the statement tree; one _stmt_<Kind> method per IR
     |           statement (Tile, Load, Write, Assign, Accum, Loop,
     |           StridedLoop, Cond, Smem, Sync, WarpShuffle, TreeHalve,
     |           CpAsync*, Select, Let, Init)
     |-- index.py        integer index math: affine normalization,
     |                   constant folding, reuse of computed values
     |-- elementwise.py  values, dtype conversions, arithmetic and math
     |                   functions (exp, tanh, erf, ...) built from PTX
     |-- dtypes.py       how each IR dtype lives in registers and memory
            |
            v
  ptx.py         virtual registers, labels, instruction lines, and the
                 final module text (.version / .target / entry / .reg)
            |
            v
          PTX on stdout
```

Each IR name (axis variable, SSA value, accumulator) maps to a `Value`: a PTX register plus
its dtype. Every new value gets a fresh virtual register; `ptxas` does the real register
allocation. Accumulators are the exception: they keep one register for their whole life and
are updated in place, which is what makes loops and cross-thread combines work.

## What is supported

Statements: `Tile` (with tail guard, aux threads and CTA rasterization), `Load` / `Write`
(global and shared, scalar or grouped as v2 / v4 vector accesses, atomic add via `red`),
`Assign`, `Let`, `Init`, `Accum` (add, multiply, max, min, with or without `base`), `Select`,
`Loop`, `StridedLoop` (including `end`), `Cond` (with else), `Smem` (static, or carved out of
the dynamic pool when `meta.json` asks for dynamic shared memory), `Sync` (CTA, warp, named
barriers), `WarpShuffle` and `TreeHalve` with arbitrary multi-component combine programs
(online softmax), `CpAsyncCopy` / `CpAsyncCommit` / `CpAsyncWait` (with emmy's XOR swizzle).

Elementwise ops: add, subtract, multiply, divide, maximum, minimum, negative, abs, square,
reciprocal, sqrt, rsqrt, exp, log, sin, cos, tanh, erf, sigmoid, relu, pow, where, copy,
comparisons, logical and / or / not, plus the integer ops (floor_divide, remainder, shifts,
bitwise and / or / xor, popcount).

Data types: f32, f16, f64, i32, i64 (plus u8 / u16 / u32 / u64 / i16 / bool), following
emmy's typing rules: integer arithmetic stays integer, all-f16 stays f16, everything else is
computed in f32.

Anything else (the tensor-core statements, TMA, which TASK.md says appear in no problem)
stops with a clear `compile.py: unsupported Kernel IR statement: ...` message on stderr and a
non-zero exit, never with wrong PTX.

## Design decisions

* **Follow the IR exactly, optimize only below it.** The IR already contains the schedule
  (tiling, cooperative reductions, pipelining). The compiler reproduces its semantics, including
  emmy's typing and accumulator seeding rules, and only optimizes how each step is spelled in
  PTX. This is why it generalizes: nothing depends on problem names or shapes.
* **The launch contract comes from `meta.json`.** Parameter order, block size, whether a tail
  guard is needed (`grid * block != N`) and dynamic shared memory are all read from it.
* **Exact where the reference is exact.** Arithmetic uses `.rn` instructions, which are
  correctly rounded and are never contracted into fma, so pointwise results match numpy bit for
  bit (several problems use `rtol = atol = 0`). Transcendentals use the hardware approximations
  with care at the edges (range reduction for sin / cos, Taylor series near zero for tanh and
  erf); measured errors are around 1e-7 relative.
* **Index math is normalized, not just emitted.** Every integer index becomes
  `sum(coeff * atom) + constant`. The constant turns into an immediate address offset
  (`[%rd10+12]`), terms like `(a + b) - a` cancel, and equal expressions are computed once.
  Reordering is exact because s32 arithmetic is arithmetic modulo 2^32.
* **Three optimizations, each small and explainable:** loops the IR marks `unroll=True` are
  fully unrolled (the counter becomes a constant, so the matmul inner loop is pure loads and
  fmas with immediate offsets); `acc += a * b` becomes one `fma`; inputs the kernel never
  writes are loaded through the read-only cache (`ld.global.nc`) and each element is loaded once.
  Together these halved the 512x512 matmul (29.3 us to 15.0 us).
* **Tested like the hidden set.** Variants are generated by the organizers' generator, run on
  real hardware, and compared with float64 numpy references.

## Repository layout

```
compile.py            the CLI entry point (the submission contract)
emmyptx/              the compiler (see Architecture)
tests/open/           the 15 open problems (copied from the hackathon repo)
tests/variants.py     torch expressions for the generated variants
tests/generated/      the generated variants (IR, meta.json, expected.npz)
tests/test_compiler.py offline checks (pytest)
tools/run_all.py      run everything on the remote runner
tools/runner.py       runner client with a response cache
tools/gen_variants.py regenerate tests/generated/
LEARN.md              how the compiler works, from zero GPU knowledge
VIDEO_SCRIPT.md       outline for the walkthrough video
RESULTS.md            the pass table and timings
```
