# emmyptx: Emmy Kernel IR to PTX

A compiler for the [Emmy Hackathon 2026](https://riftstack.ai/hackathon). It reads one
kernel of Emmy's Kernel IR (json) and prints NVIDIA PTX that `ptxas` assembles directly:

```bash
pip install emmy-ml==0.3.11 numpy
python compile.py path/to/kernel.ir.json > out.ptx
```

The IR to PTX lowering is entirely our own Python (about 2,200 lines in `emmyptx/`). Emmy is
used only to deserialize the IR (`Graph.from_dict`); no emmy renderer, nvcc, NVRTC, LLVM or
other compiler is involved. `ptxas` only ever runs on the hackathon's remote runner.

## Results

The original 193 problems pass on the runner's RTX 5090. The expanded suite has 248 problems:
15 open problems, 227 generated variants and 6 synthetic IRs. Two new transformer cases
time out and remain in the suite as failures. Fresh results and limitations are recorded in
[RESULTS.md](RESULTS.md) and [the GPU report](docs/gpu_results.md).

| category | open set | generated variants | synthetic IR |
|---|---|---|---|
| 1 pointwise | 4/4 | 69/69 | 4/4 |
| 2 reductions | 4/4 | 46/46 | 2/2 |
| 3 matmul | 2/2 | 41/41 | - |
| 4 fused | 3/3 | 47/47 | - |
| 5 attention-like | 2/2 | 22/24 | - |

The generated variants are our stand-in for the hidden set: 227 packaged torch expressions with new
shapes (odd sizes, sizes that do not divide the block, tiny and large), new ops (exp, log,
tanh, sigmoid, silu, gelu, erf, sin, cos, sqrt, rsqrt, pow, where, clamp, max/min, abs),
broadcasting, f16 / f64 / int32 / int64 data, other reduction axes and block sizes, schedules that
split a reduction across blocks with atomics, matmuls with odd K, MLPs, layernorm, softmax,
variance, and attention of several shapes and head sizes. They are traced and packaged by the
organizers' own `tools/generate.py`, exactly like the hidden set. The synthetic IRs
(`tests/synthetic.py`) are built from emmy's statement classes for forms the generator never
produced in our runs (Cond with else, segmented TreeHalve, aux threads, 4/8-byte cp.async, ...),
so every statement kind in the compiler has run on the GPU at least once.

The DL extension probes 76 expressions at 8x32 and odd 3x17 shapes. It packages 55, including
embedding gather, both halves of RoPE, affine norms, softmax, decomposed log-softmax, dense
cross-entropy, SwiGLU, GELU and deeper MLPs, KV attention, pooling reductions, inference
BatchNorm, channel affine transforms, cumsum and cosine similarity. Convolution and ViT
patch cases start from prepared windows or patches; they test the matmul stage. The
[coverage report](docs/DL_COVERAGE.md) lists every attempted expression, generation error
and GPU failure. Unsupported generator cases do not count as passing tests.

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
`python tools/gen_variants.py --hackathon ../emmy-hackathon-2026`. The synthetic IRs are
rebuilt with `python tests/synthetic.py`.

## Architecture

The optional [NVRTC comparison](docs/bench/RESULTS_nvcc.md) and [local demo](#demo)
live in `bench/` and are not imported by the submission compiler.

```
  kernel.ir.json + meta.json
            |
            v
  loader.py      load the IR with Graph.from_dict, buffers (shape, dtype),
                 parameter order and launch geometry from meta.json
            |
            v
  analysis.py    whole-kernel facts: read-only inputs, accumulators,
                 products that can fuse into an fma, loads that can
                 merge into one vector load
            |
            v
  lower.py       walk the statement tree; one _stmt_<Kind> method per IR
     |           statement: Tile, Assign, Let, Select, Init, Accum, Loop,
     |           StridedLoop, Cond, Sync
     |-- memory.py       Smem, Load, Write, CpAsync*: addressing, vector
     |                   accesses, alignment proofs
     |-- combine.py      WarpShuffle and TreeHalve with their combine programs
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
computed in f32. Integer `//` and `%` truncate toward zero, as in the C that emmy itself
generates (so they differ from Python's floor semantics for negative operands, exactly as
emmy's own kernels do).

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
  bit (several problems use `rtol = atol = 0`). Transcendentals are built like libm: exp with
  Cody-Waite reduction and an exact 2^n, sin / cos with quadrant reduction and polynomials,
  tanh and erf with Taylor series near zero. Measured on the GPU they are within 1e-7 to 2e-7
  relative (table in RESULTS.md).
* **Index math is normalized, not just emitted.** Every integer index becomes
  `sum(coeff * atom) + constant`. The constant turns into an immediate address offset
  (`[%rd10+12]`), terms like `(a + b) - a` cancel, and equal expressions are computed once.
  Reordering is exact because s32 arithmetic is arithmetic modulo 2^32.
* **Optimizations, each small, generic and measured:** loops the IR marks `unroll=True` are
  fully unrolled (the counter becomes a constant, so the matmul inner loop is pure loads and
  fmas with immediate offsets); other innermost constant-trip loops are unrolled by 4; a
  thread's strided loop runs blocks of up to 16 in-range iterations so many loads are in flight
  (same order, so sums are bit-identical); `acc += a * b` becomes one `fma`; inputs the kernel
  never writes use the read-only cache (`ld.global.nc`) and each element is loaded once;
  neighbouring aligned f32 loads become one `v2` / `v4` load. Every step was A/B measured on the
  runner and kept only where it helped; rejected variants are listed in RESULTS.md. Examples:
  512x512 matmul 29.3 -> 15.0 us, single-block sums about 2x faster, several attention kernels
  1.6 to 5x faster.
* **Robust by construction.** `cp.async` needs aligned addresses; the compiler proves alignment
  with a divisibility analysis of the normalized index and loop strides, and where it cannot,
  emits a run-time check with an element-wise fallback, so it never produces a faulting copy.
  Integer identities saturate like `cvt`, integer max / min are exact, and any unsupported
  construct stops with a clear error.
* **Tested like the hidden set.** Variants are generated by the organizers' generator, run on
  real hardware, and compared with float64 numpy references.

## Repository layout

```
compile.py            the CLI entry point (the submission contract)
emmyptx/              the compiler (see Architecture)
tests/open/           the 15 open problems (copied from the hackathon repo)
tests/variants.py     torch expressions for the generated variants
tests/generated/      the generated variants (IR, meta.json, expected.npz)
tests/synthetic.py    hand-built IR for forms emmy rarely emits
tests/synthetic/      those problems (IR, meta.json, expected.npz)
tests/test_compiler.py offline checks (pytest)
tools/run_all.py      run every suite on the remote runner
tools/runner.py       runner client with a response cache
tools/gen_variants.py regenerate tests/generated/
LEARN.md              how the compiler works, from zero GPU knowledge
VIDEO_SCRIPT.md       outline for the walkthrough video
RESULTS.md            the pass table and timings
```

## Comparison with nvcc

This machine has no nvcc executable. The comparison uses **Emmy + NVRTC 12.8**, compiling
Emmy's CUDA for the same scheduled IR to `compute_89` PTX. It does not establish actual
nvcc performance. See [the measurements](docs/bench/RESULTS_nvcc.md),
[execution size sweep](docs/bench/size_sweep.png) and [compile times](docs/bench/compile_times.png).
Both plots also have SVG copies in the same folder.

Use Python 3.12 and the organisers' checkout at `../emmy-hackathon-2026`:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r bench/requirements.txt
python bench/compare_nvcc.py
python bench/compare_nvcc.py --resume --reuse-generated
```

Compile measurements include fresh processes, warm IR compilation and pure lowering.
GPU measurements use identical launches and correctness checks, without response caching.
The 2^24-element pointwise cases exceed the runner's output-return limit and are recorded
as skips. Measurements include raw samples, versions and IR/source hashes.

## Demo

The full pre-norm transformer demo uses sequence length 8, model width 32 and feed-forward
width 64. It traces eight small stage models, runs their compiled kernels in order, and
checks the final result against PyTorch running the same whole block. The causal mask is
an additive input tensor. Host NumPy arrays carry results between stages.

```bash
python bench/demo_transformer.py --backend local
python bench/demo_transformer.py --backend remote
python bench/demo_transformer.py --backend local --reuse-generated
```

Both GPUs pass, with maximum absolute errors around 2e-7. Results and exact console output
are saved in [demo_transformer_result.json](docs/demo_transformer_result.json).
No compiler changes or external CUDA compiler are used by this demo. The original MLP
demo is also available:

```bash
python bench/demo_e2e.py
python bench/demo_e2e.py --backend local --reuse-generated
```

This defines a PyTorch MLP with widths 32, 64 and 10, batch size 8, and traces it through
Emmy using the organisers' generator. Two kernels compiled by `compile.py` run in order,
sharing the hidden-layer buffer on the RTX 3050 Laptop GPU. Hidden activations and logits
are compared with PyTorch at rtol=1e-3, atol=1e-4. The script prints errors and kernel timings.

Only the demo's in-memory PTX target is rewritten from sm_89 to sm_86 before driver loading.
If local loading is unavailable, auto mode reports the cause and uses the remote runner.
Numerical mismatches are failures. [Recorded demo result](docs/demo_result.json).

## Regenerating the DL probes

```bash
python tools/gen_variants.py --contains _dl_ --report docs/dl_generation.json
python tools/gen_variants.py --contains _dl_ --report docs/dl_generation.json --resume
```

The command returns nonzero for the documented unsupported probes, while continuing through
all cases. Invalid packages are retained under `.cache/rejected/` with their errors in the
report. Windows uses a small CLI adapter for Emmy's POSIX-only import and UTF-8 dumps;
the organisers' tracing and packaging functions are unchanged.
