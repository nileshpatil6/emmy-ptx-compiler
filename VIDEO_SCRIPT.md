# Video script: 4 minutes 40 seconds

Use Python 3.12. Open README, the light-theme benchmark plots and a terminal.
Use recorded JSON for the large suite, and run the small demo live. Describe
cached suite output as cached.

Run the transformer demo once without `--reuse-generated` before recording to prepare
its stage IRs.

## 0:00 to 0:25: What the compiler does

Screen: README and compile.py.

"EmmyPTX translates Emmy's scheduled Kernel IR directly to NVIDIA PTX. The lowering
is our Python code. The submission compiler does not use CUDA C++, nvcc or NVRTC."

"The original 193 GPU problems still pass. The expanded suite has 55 more problems,
with two transformer timeouts that I have kept in the results."

## 0:25 to 1:05: One kernel from input to output

Screen: the scalar-add IR, then:

```bash
python compile.py tests/open/p00_scalar_add/k_add_pointwise.ir.json
```

Show the Tile, loads, add and write. Point out parameter order, thread index,
tail guard, rounded add and vector store. Meta.json supplies the launch.

## 1:05 to 1:50: Architecture and one GPU idea

Screen: README architecture and row-sum IR.

Explain loading, analysis, statement lowering and PTX text. Each thread sums
a strided slice, then warp shuffles and a shared-memory tree combine partials.
The same combine-program mechanism supports max and online softmax.

Briefly show affine index simplification and an unrolled matmul loop. These
are general transformations below the schedule, without problem-name checks.

## 1:50 to 2:30: DL coverage and limits

Screen: docs/DL_COVERAGE.md and the fresh GPU table.

Show embedding, both RoPE halves, norms, SwiGLU, GELU, KV attention, pooling
reductions and cumsum. Fixtures come from the organisers' generator, at regular
and odd shapes. Convolution cases use prepared windows: the claim is the matmul
stage, not a complete native convolution.

Show a frontend failure and transformer timeout. Distinguish tracing, packaging
and execution failures. Mention the small integer-index cast fix and wishlist.

## 2:30 to 3:10: NVRTC comparison, 40 seconds

Screen: docs/bench/size_sweep.png, then compile_times.png.

"This machine has no nvcc executable, so this baseline is Emmy's CUDA compiled
with NVRTC 12.8. Both paths use the same scheduled IR and launch geometry. It
is not an actual nvcc measurement."

Read one faster and one slower case from the measured table. Explain the axes,
then fresh-process and preloaded compile measurements. GPU values are medians,
clocks are not locked, and near-equal values can be noise. Mention runner-limit
skips. Do not claim a cause for speed differences without profiling.

## 3:10 to 3:50: Live model demo, 40 seconds

Screen: terminal.

```bash
python bench/demo_transformer.py --backend local --reuse-generated
```

"This is a full pre-norm transformer block: LayerNorm, causal attention with
Q/K/V projections, a residual, another LayerNorm, a SwiGLU MLP and a final
residual. Small stage models are traced separately to avoid Emmy's bad fusion.
Our compiled kernels run on this RTX 3050, with host arrays between stages."

Point at the stage lines, whole-block PyTorch errors and PASS. The additive
causal mask is an input tensor. Only the demo's loaded
PTX target changes to sm_86; the submission still emits sm_89. Compile and
kernel times are separate, not total model latency.

## 3:50 to 4:40: Reproducibility and close

Screen: RESULTS and the implementation report.

Show before/after categories, test commands, raw samples, environment versions
and failures. Read the final pytest count and GPU totals from the report.
Do not describe the entire suite as green.

"The compiler stays independent, measurements include wins and losses, and
the remaining unsupported cases are documented for review."
