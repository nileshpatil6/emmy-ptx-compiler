# LEARN: how this compiler works, from zero GPU knowledge

This guide assumes you know Python and have built a compiler before (English to Python), but
have never written GPU code or PTX. Read it top to bottom once; afterwards each section works
as a reference. Every PTX snippet below is real output of `python compile.py ...`.

---

## 1. The big picture

Emmy is a system that turns PyTorch programs into GPU kernels. Internally it goes:

```
PyTorch expression -> ... -> Kernel IR -> CUDA C++ -> (nvcc) -> PTX -> (ptxas) -> GPU binary
```

The hackathon asks us to cut out the middle: go **from Kernel IR straight to PTX**. PTX is
NVIDIA's assembly-like language. A tool called `ptxas` (run on the organizers' server) turns
PTX into the real machine code. Skipping CUDA C++ and nvcc makes compilation much faster,
which matters because Emmy's autotuner compiles thousands of kernel variants.

So our program is a classic compiler:

* **input**: a tree of statements (the Kernel IR, saved as json),
* **output**: text in another language (PTX),
* **contract**: `python compile.py kernel.ir.json > out.ptx`.

Same shape as your English to Python compiler: parse (done for us by emmy's loader), walk the
tree, emit target code for each node.

---

## 2. GPU basics in five minutes

### Threads, warps, blocks, grids

A CPU runs a few threads fast. A GPU runs **thousands of threads at once**, each slower, all
executing the same program (the *kernel*) on different data.

* A **thread** is one running copy of the kernel. It has its own registers.
* 32 threads form a **warp**. A warp executes one instruction at a time for all 32 threads
  together (like 32 people doing the same dance move). Threads in a warp can swap register
  values directly (*shuffles*).
* Threads are grouped into **blocks** (here 32 to 512 threads). Threads of one block run on
  the same processor, can share a small fast memory (**shared memory**) and can wait for each
  other (**barrier**, `bar.sync`).
* All blocks together form the **grid**.

When a kernel launches, the launcher says "grid of G blocks, each of B threads". Each thread
can ask "which thread am I?":

* `%tid.x`   = my index inside my block (0 .. B-1)
* `%ctaid.x` = my block's index in the grid (0 .. G-1) (CTA = "cooperative thread array" = block)
* `%ntid.x`  = B, the block size

The usual global thread number is `gid = ctaid * ntid + tid`.

In this hackathon the grid and block sizes are fixed by `meta.json`; we never choose them.

### Memories

| memory | who sees it | speed | PTX state space |
|---|---|---|---|
| registers | one thread | fastest | `.reg` |
| shared memory | one block | very fast | `.shared` |
| global memory | everyone, the CPU too | slow (hundreds of cycles) | `.global` |

Inputs and outputs live in global memory. The game of GPU programming is to read global
memory as little and as efficiently as possible: neighbouring threads should read neighbouring
addresses (*coalescing*), and data that is reused should be staged in shared memory or
registers.

---

## 3. What Kernel IR is

Open `tests/open/p00_scalar_add/k_add_pointwise.ir.txt` (the pretty-printed IR):

```
Tile[a0] (N=64)
    in0__u0 = load x0[((a0 * 2) + 0)]
    in1__u0 = load x1[((a0 * 2) + 0)]
    f32 v0__u0 = add(in0__u0, in1__u0)
    in0__u1 = load x0[((a0 * 2) + 1)]
    in1__u1 = load x1[((a0 * 2) + 1)]
    f32 v0__u1 = add(in0__u1, in1__u1)
    add[((a0 * 2) + 0)] = (v0__u0, v0__u1)
```

Read it as: "There are 64 *cells* of work, numbered by `a0`. Each cell loads two elements of
`x0` and `x1`, adds them, and writes two results." The problem is `x0 + x1` on 128 floats; the
IR already decided that each thread handles 2 elements (that is the `__u0`, `__u1` unrolling),
and that the two results are stored together.

The IR is a tree of **statements** (the full list is in TASK.md):

* `Tile` maps the cells onto threads (one thread per cell).
* `Load name = buffer[index]` and `Write buffer[index] = value` touch memory.
* `Assign name = op(args)` is arithmetic (add, exp, ...). Names are SSA: each defined once.
* `Accum acc <- add(acc, v)` folds a value into an accumulator inside a loop.
* `Loop` / `StridedLoop` / `Cond` are control flow.
* `Smem`, `Sync`, `WarpShuffle`, `TreeHalve`, `CpAsync...` are the GPU-specific machinery.

Index expressions like `((a0 * 2) + 0)` are small expression trees (`Var`, `Literal`,
`BinaryExpr`, `TernaryExpr`).

The other input is `meta.json`: the launch (grid 1, block 256 for p00), the parameter order
(`arg_order`: x0, x1, add) and the tolerance. Note p00 launches 256 threads for 64 cells: the
extra 192 threads must do nothing. That guard is not in the IR; we derive it.

We load the IR with emmy's own `Graph.from_dict`, which hands us typed Python objects
(`Tile`, `Load`, ...). Our compiler walks those objects.

---

## 4. What PTX looks like

PTX is a typed assembly language with unlimited *virtual* registers:

```
.version 8.0                 // PTX language version
.target sm_89                // GPU generation (Ada); runs on newer GPUs too
.address_size 64

.visible .entry k_add_pointwise(      // the kernel, name must match the IR
	.param .u64 param_x0,              // buffers arrive as 64-bit addresses
	.param .u64 param_x1,
	.param .u64 param_add
)
{
	.reg .f32   %f<7>;                 // declare registers %f1..%f6 (32-bit floats)
	.reg .b32   %r<8>;                 // 32-bit integers
	.reg .b64   %rd<13>;               // 64-bit integers (addresses)
	.reg .pred  %p<2>;                 // predicates (true / false)
	...
	ret;
}
```

Instruction format: `opcode.type destination, source, source;` for example
`add.rn.f32 %f3, %f1, %f2;` means `f3 = f1 + f2` in 32-bit float with round-to-nearest.

Control flow uses **predicates** and **labels**:

```
	setp.ge.s32  %p1, %r6, 64;     // p1 = (r6 >= 64), signed 32-bit compare
	@%p1 bra     $L_exit_1;        // if p1: jump to label $L_exit_1
```

Memory instructions name the state space: `ld.global.f32`, `st.shared.f32`, and
`ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd5]` loads four floats at once (a *vector* load).
Addresses can carry a constant offset: `[%rd10+12]` is "the address in %rd10 plus 12 bytes".

---

## 5. Lowering p00 line by line

Here is our complete output for p00 (`python compile.py tests/open/p00_scalar_add/*.ir.json`),
with the IR statement each part comes from:

```
	// buffer parameters -> global addresses          (prologue, from meta.json arg_order)
	ld.param.u64        %rd1, [param_x0];              read the pointer argument
	cvta.to.global.u64  %rd2, %rd1;                    make it a global-space address
	... same for x1 (%rd4) and add (%rd6) ...
	// thread coordinates
	mov.u32   %r1, %tid.x;                             thread index in the block
	mov.u32   %r2, %ctaid.x;                           block index
	mov.u32   %r3, %ntid.x;                            block size
	and.b32   %r4, %r1, 31;                            lane = tid % 32   (for reductions)
	shr.u32   %r5, %r1, 5;                             warp = tid / 32
	// Tile[a0] (N=64)
	mad.lo.s32  %r6, %r2, %r3, %r1;                    gid = ctaid * ntid + tid   (a0 = gid)
	setp.ge.s32 %p1, %r6, 64;                          the guard: 256 threads, 64 cells
	@%p1 bra    $L_exit_1;
	mul.lo.s32  %r7, %r6, 2;                           a0 * 2
	mul.wide.s32 %rd7, %r7, 4;                         byte offset = index * sizeof(float)
	add.s64     %rd8, %rd2, %rd7;                      address of x0[a0*2] = base + offset
	ld.global.nc.f32 %f1, [%rd8];                      in0__u0 = load x0[a0*2 + 0]
	add.s64     %rd9, %rd4, %rd7;                      address of x1[a0*2] (same byte offset %rd7)
	ld.global.nc.f32 %f2, [%rd9];                      in1__u0 = load x1[a0*2 + 0]
	add.rn.f32  %f3, %f1, %f2;                         v0__u0 = add(in0__u0, in1__u0)
	ld.global.nc.f32 %f4, [%rd8+4];                    in0__u1 = load x0[a0*2 + 1]  (same register, +4 bytes)
	ld.global.nc.f32 %f5, [%rd9+4];                    in1__u1 = load x1[a0*2 + 1]
	add.rn.f32  %f6, %f4, %f5;                         v0__u1
	add.s64     %rd10, %rd6, %rd7;                     address of add[a0*2]
	st.global.v2.f32 [%rd10], {%f3, %f6};              add[a0*2] = (v0__u0, v0__u1), one vector store
$L_exit_1:
	ret;
```

Things to notice, each a deliberate decision:

* **The guard** comes from comparing the launch (`grid * block = 256` threads, from meta.json)
  with the Tile's N (64). When they are equal (p02: 4 x 256 = 1024 = N) no guard is emitted.
* **`+0` / `+4` offsets**: the compiler splits every index into "variable part + constant"
  (`a0*2` and `+1`), computes the variable part once, and puts the constant into the address.
* **`.nc`** (non-coherent, read-only cache) is used for inputs the kernel never writes.
* **`.rn`** makes the add correctly rounded and prevents fusing it with a multiply, so the
  result equals numpy's exactly (p00 has tolerance 0).
* **The grouped write** `(v0__u0, v0__u1)` becomes one `st.global.v2.f32`.

---

## 6. How a reduction works (r01: sum of 1024 floats)

The IR (`tests/open/r01_array_sum/k_sum_1_reduce.ir.txt`):

```
Tile[a0_co] (N=128)                      one block of 128 cooperating threads
    for a0 in a0_co..1024:128            StridedLoop: thread t sums x[t], x[t+128], x[t+256], ...
        in0 = load x[a0]
        f32 acc0 <- add(acc0, in0)       Accum: acc0 starts at 0 (the identity of add)
    WarpShuffle(acc0, length=32)         combine the 32 partial sums of each warp
    Smem float acc0_smem[4]              4 warps -> 4 slots of shared memory
    if ((lane == 0)):
        acc0_smem[warp] = acc0           first thread of each warp publishes its warp's sum
    Sync                                 wait until all 4 are written
    TreeHalve(acc0, length=4, tid=warp)  combine the 4 slots: 4 -> 2 -> 1
    if ((a0_co == 0)):
        sum_1[] = acc0                   thread 0 writes the answer
```

Three levels: each **thread** sums its share serially, each **warp** combines its 32 threads
with shuffles, the **block** combines its warps through shared memory.

### Step 1: strided loop with an accumulator

```
	mov.f32   %f1, 0f00000000;              acc0 = 0.0  (the Accum's seed, placed before its loop)
	mov.f32   %f2, %f1;
	mov.s32   %r7, %r6;                     a0 = a0_co (this thread's start)
$L_a0_top_2:
	setp.ge.s32 %p1, %r7, 1024;             while a0 < 1024
	@%p1 bra  $L_a0_done_3;
	... address of x[a0] ...
	ld.global.nc.f32 %f3, [%rd6];
	add.rn.f32 %f2, %f2, %f3;               acc0 += in0, in place in the same register
	add.s32   %r7, %r7, 128;                a0 += 128
	bra       $L_a0_top_2;
$L_a0_done_3:
```

Why stride by 128 and not give each thread a contiguous chunk? In each round, threads 0..127
read x[0..127] together: neighbouring threads, neighbouring addresses, so the hardware merges
them into few wide memory transactions (coalescing).

An accumulator is the one kind of value that keeps **one register for its whole life**. Every
other value gets a fresh register. That is why `%f2` appears on both sides of the add.

### Step 2: the warp shuffle butterfly

`shfl.sync.bfly.b32 d, a, k` gives each thread the value `a` held by the thread whose lane
number is `mine XOR k`. With k = 16, 8, 4, 2, 1:

```
	activemask.b32      %r8;                     which lanes are active (all 32 here)
	shfl.sync.bfly.b32  %f4, %f2, 16, 31, %r8;   partner = lane ^ 16
	add.rn.f32          %f2, %f2, %f4;           now lane i holds x_i + x_(i^16)
	shfl.sync.bfly.b32  %f5, %f2, 8, 31, %r8;
	add.rn.f32          %f2, %f2, %f5;           sums of 4
	... 4, 2, 1 ...                              after 5 steps every lane has the warp's total
```

Picture 8 lanes (steps 4, 2, 1): after step 4, lane 0 has 0+4, lane 1 has 1+5 ...; after step 2,
lane 0 has 0+4+2+6; after step 1 it has all eight. 32 values are summed in 5 steps, with no
memory traffic at all, only register exchanges.

The IR does not hard-code "add": `WarpShuffle` carries a little **combine program** (a list of
`Assign`s). For a sum it is one add; for online softmax it is ten instructions (section 7). We
run that program after every shuffle, so one implementation handles every reduction.

### Step 3: shared memory and the tree

```
	// if (lane == 0)
	setp.eq.s32  %p2, %r4, 0;
	@!%p2 bra    $L_endif_4;                       skip unless lane == 0
	mov.u32      %r9, smem_acc0_smem;              address of the shared array
	mad.lo.s32   %r10, %r5, 4, %r9;                + warp * 4 bytes
	st.shared.f32 [%r10], %f2;                     acc0_smem[warp] = acc0
$L_endif_4:
	bar.sync     0;                                Sync: every thread waits here
```

`TreeHalve(length=4)` then halves: threads with `warp < 2` add slot `warp+2` into slot `warp`,
barrier, threads with `warp < 1` add slot 1 into slot 0, barrier, and finally every thread reads
slot 0. Our compiler unrolls the halving loop at compile time because `length` is a constant.

Why not just let every warp do `atomicAdd` on the output? Atomics on one address serialize;
the tree is deterministic and fast.

---

## 7. Fused kernels: one reduction plus a pointwise pass

RMSNorm (f01) is "sum of squares per row, then scale every element". The IR is the r04 row
reduction followed by a second `StridedLoop` that reads the row again and writes
`x * rsqrt(mean + eps) * w`. Nothing new is needed: the reduction leaves the result in every
thread's `acc0` (TreeHalve broadcasts it), and the second loop just uses it.

Softmax (f02) is the interesting one: it needs the row max *and* the sum of `exp(x - max)`.
Emmy uses the **online softmax** trick: keep two coupled accumulators, the running max `m` and
the running sum `s`, and when a bigger value arrives rescale the sum:

```
new_m = max(m, x)
s     = s * exp(m - new_m) + exp(x - new_m)
m     = new_m
```

In the IR this appears as `Accum` statements plus `Assign`s (`acc1 <- add(sa, sb)` is an
Accum with a `base`), and `WarpShuffle(acc0, acc1)` / `TreeHalve(acc0, acc1)` carry a 10-step
combine program that merges two (max, sum) pairs. Because our WarpShuffle and TreeHalve just
run whatever combine program the IR provides, softmax needed no special case.

Detail that matters: the identity of `max` is `-1e30` (emmy's choice), not minus infinity, so
`exp(-1e30 - x)` is 0 and never NaN.

---

## 8. How matmul tiling works (m01: 512 x 512 x 512)

Naive matmul: each output `C[i][j]` reads a whole row of A and a whole column of B from global
memory. Every element of A is read 512 times from slow memory. Tiling fixes this:

1. A block of 512 threads owns a 32 x 64 tile of C.
2. Loop over K in steps of 64. In each step the block copies a 32 x 64 slab of A and a
   64 x 64 slab of B into **shared memory**, together, once.
3. Each thread then computes a 2 x 2 micro-tile of C from shared memory: for each of the 64
   k values it loads 2 values of A and 2 of B and does 4 multiply-adds. Each loaded value is
   used twice (register reuse).

The m01 IR adds two refinements, and our compiler simply follows them:

* **cp.async**: a copy from global to shared memory that runs in the background
  (`cp.async.cg.shared.global [dst], [src], 16;` copies 16 bytes). `cp.async.commit_group`
  closes a batch, `cp.async.wait_group N` waits until at most N batches are still in flight.
* **A 3-stage pipeline**: the shared arrays hold 3 slabs (`_a_smem[96, 64]` = 3 x 32 rows).
  While the threads compute on slab k, the copies for slab k+2 are already flying. The index
  `((_ks / 64) + 2) % 3` picks the ring-buffer slot.

The inner `for _ki in 0..64` loop is marked `unroll=True`. We unroll it completely: the
compiler emits the body 64 times with `_ki` bound to the constants 0..63. Then every shared
memory address becomes "one register + a constant", and the loop is pure loads and fmas:

```
	ld.shared.f32  %f9,  [%r48+24576];      B[_ki=0][col]
	ld.shared.f32  %f10, [%r53];            A[row][_ki=0]
	fma.rn.f32     %f2, %f9, %f10, %f2;     acc00 += b0 * a0
	ld.shared.f32  %f11, [%r48+24580];      B[0][col+1]
	fma.rn.f32     %f4, %f11, %f10, %f4;    acc01 += b1 * a0
	ld.shared.f32  %f12, [%r53+256];        A[row+1][0]
	fma.rn.f32     %f6, %f9, %f12, %f6;     acc10
	fma.rn.f32     %f8, %f11, %f12, %f8;    acc11
	ld.shared.f32  %f13, [%r48+24832];      _ki = 1: same registers, new offsets
	...
```

`fma` (fused multiply-add) does `a * b + c` in one instruction. The IR writes it as a
`multiply` followed by an `Accum add`; our analysis pass notices the product is used exactly
once, by that accumulation, and fuses them. Unrolling plus fma fusion halved the m01 time.

m02 shows **split-K with atomics**: the K dimension is split across blocks, each block adds its
partial result into the output with `red.global.add.f32` (an atomic add), and `meta.json` lists
the output under `zero_outputs` so the runner zeroes it before each launch.

---

## 9. Attention-like kernels (a01)

`softmax(q @ k^T / 8) @ v` in one kernel. Each thread owns a few output rows and columns. For
every key position `a1` it computes the scores `q[row] . k[a1]` (an inner `Loop` with an Accum),
then updates online-softmax accumulators exactly as in section 7, and at the same time
accumulates `v[a1] * exp(...)` with the same rescaling. At the end it divides by the softmax
sum. In compiler terms it is a combination of things we already have: nested `Loop`s, Accums
with `base`, Assigns with exp, and grouped loads. No new statement kinds.

---

## 10. The compiler's stages and files

```
compile.py               CLI: read argv[1], print PTX or an error on stderr
emmyptx/loader.py        Stage 1: Graph.from_dict, buffers, meta.json -> KernelSpec
emmyptx/analysis.py      Stage 2: read-only inputs, accumulators, fusable products
emmyptx/lower.py         Stage 3: one _stmt_<Kind> method per statement kind
emmyptx/index.py           integer index expressions (normalize, fold, reuse)
emmyptx/elementwise.py     Value, conversions, arithmetic, exp / tanh / erf ...
emmyptx/dtypes.py          how f32 / f16 / i64 ... live in registers and memory
emmyptx/ptx.py           Stage 4: registers, labels, final module text
```

A tour in the order a kernel flows through:

1. **`loader.load_kernel`** returns a `KernelSpec`: kernel name, the statement list, the
   parameter order (from meta.json), every buffer's shape and dtype, and the launch geometry.
2. **`analysis`** answers three questions before emitting anything (see the file).
3. **`lower.KernelLowering.run`** emits the prologue (load the pointer parameters, read
   `%tid.x` and friends, compute `lane` and `warp`), then calls `_body` on the statements.
   `_body` dispatches by class name: a `Load` goes to `_stmt_Load`, and so on. The
   environment `env` maps every IR name to a `Value(reg, dtype)`.
4. **Memory**: `_flatten` turns a multi-dimensional index into a row-major element offset
   (`x[a0, a1]` with shape (64, 512) becomes `a0 * 512 + a1`), `_address` turns that into
   `[register+bytes]`.
5. **`index.IndexLowerer`** lowers integer expressions. Its key idea is `linearize`: any
   integer expression becomes `sum(coefficient * atom) + constant`, where atoms are variables
   or non-linear pieces like `x / 64`. This cancels terms, folds constants, and gives every
   expression a canonical form, so a cache can reuse values already computed in the current
   block (that cache is scoped: values computed inside a loop body are forgotten when the
   body ends, and a value is forgotten when a name it reads is reassigned).
6. **`elementwise.Elementwise`** emits the arithmetic. It follows emmy's type rules
   (`dtypes.promote`) and builds the math functions from PTX's hardware approximations.
7. **`ptx.render_kernel`** wraps the instruction lines: header, `.shared` declarations, entry
   signature, `.reg` declarations sized from the register counters.

### Numerics in one paragraph

PTX has `ex2.approx` (2^x), `lg2.approx` (log2), `sin.approx`, `cos.approx`, `rsqrt.approx`
and exact `div.rn`, `sqrt.rn`, `rcp.rn`. So `exp(x) = ex2(x * log2(e))`, `log(x) =
lg2(x) * ln(2)`; sin and cos first reduce the argument to [-pi, pi] (the hardware is only
accurate there); tanh and erf use a Taylor series near zero (where `1 - 2/(e^2x + 1)` would
lose relative precision) and the exp-based formula elsewhere. Integer and f32 values convert
with C semantics (`cvt.rzi` truncates toward zero), f16 is computed in f32 and rounded once.

---

## 11. How we test

* `python -m pytest tests`: offline, no GPU. Compiles all 110 IRs and checks the contract
  (header, entry name, parameter order, ASCII only, deterministic output) plus unit tests for
  the tricky helpers.
* `python tools/run_all.py`: the real test. It runs every problem on the runner's RTX 5090
  exactly like `submit.py` and prints a pass table per category.
* `tests/variants.py`: 95 extra problems, our imitation of the hidden set, generated by the
  organizers' own generator from torch expressions, with float64 numpy references.

---

## 12. Questions a judge might ask

**Q1. Did you use emmy's renderer, nvcc or any existing compiler?**
No. Emmy is only used to deserialize the IR (`Graph.from_dict`), which the rules require. Every
PTX instruction is produced by our Python code in `emmyptx/`. ptxas runs only on the runner.

**Q2. How do you know you will work on the hidden set and not just the 15 problems?**
Nothing in the compiler mentions a problem name or shape; it translates statement kinds. To
check generalization we generated 95 new problems with the organizers' own generator (odd
sizes, new ops, f16 / f64 / int64, other block sizes, split reductions, other matmul and
attention shapes) and all pass on the real GPU.

**Q3. Where does the launch geometry come from?**
From `meta.json` (grid, block, parameter order, dynamic shared memory). We never choose it.
We compare `grid * block` with the Tile's N to decide whether extra threads need a guard.

**Q4. Why do you not need register allocation?**
PTX has unlimited virtual registers; ptxas does the real allocation into the hardware's
registers. We give every value a fresh register, except accumulators, which keep one
register because loops update them.

**Q5. How does the warp shuffle reduction work?**
`shfl.sync.bfly` exchanges registers between lanes whose numbers differ in one bit. After the
XOR distances 16, 8, 4, 2, 1, every lane holds the combination of all 32 values (section 6).

**Q6. How did you handle softmax's two accumulators?**
We did not special-case it. WarpShuffle and TreeHalve carry a combine program of Assigns; we
lower that program after every exchange, with assignments to state names updating the state
registers in place. Sum, max and online softmax all go through the same code.

**Q7. How is your PTX exact for `rtol = 0` problems?**
`add.rn.f32` and friends are IEEE correctly rounded and `.rn` forbids ptxas from fusing a
multiply and an add into an fma (which rounds once instead of twice). So each pointwise step
rounds exactly as numpy does. We only create fmas ourselves where the IR accumulates a product
(matmul, reductions), where the tolerance allows it.

**Q8. What optimizations did you do and how much did they help?**
Full unrolling of loops the IR marks `unroll=True`, canonical affine index expressions (so
addresses become register + constant and equal expressions are computed once), fusing
`acc += a * b` into `fma`, and read-only cached loads for inputs. The 512 x 512 matmul went
from 29.3 to 15.0 microseconds. They are all generic: none depends on a specific problem.

**Q9. What happens with an IR feature you do not support?**
`compile.py` exits non-zero with `unsupported Kernel IR statement: <Kind>` (or an unsupported
op / dtype) on stderr. It never prints PTX it is not sure about.

**Q10. Did you find anything surprising?**
Yes, an emmy bug: the row variance with emmy's own split schedule produces an IR whose mean
loop forgets the split offset. Our kernel computes exactly what the IR says (verified against a
numpy model of the IR), which shows the compiler is faithful. Details in RESULTS.md.

**Q11. Why PTX `.target sm_89` when the GPU is sm_120?**
The problems are scheduled for sm_89 (RTX 4090). PTX is forward compatible: ptxas compiles
sm_89 PTX for the newer sm_120 GPU. `.version 8.0` is the PTX version sm_89 requires.

---

## 13. Glossary

* **CTA**: cooperative thread array, PTX's name for a block.
* **coalescing**: neighbouring threads accessing neighbouring addresses, merged by hardware.
* **predicate**: a 1-bit register; `@%p` executes an instruction only where it is true.
* **SSA**: static single assignment; every name is defined once.
* **state space**: where data lives (`.reg`, `.shared`, `.global`, `.param`).
* **identity**: the starting value of an accumulator (0 for add, 1 for multiply, -1e30 for max).
* **split-K**: splitting the summed dimension across blocks and combining with atomics.
