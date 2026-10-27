# Implementation report

Completed on 28 October 2026. The original 193 GPU problems still pass. The expanded
suite passes 246/248 problems, with two documented transformer timeouts. All 284
offline tests pass and Ruff is clean. No push or repository visibility change was made.

## 1. Before and after

| Category | Before | After |
|---|---:|---:|
| Pointwise | 65/65 | 77/77 |
| Reductions | 44/44 | 52/52 |
| Matmul | 31/31 | 43/43 |
| Fused | 34/34 | 50/50 |
| Attention | 19/19 | 24/26 |
| Total | 193/193 | 246/248 |

The final run used `tools/run_all.py --no-cache`, including all open, generated and
synthetic problems. It used one launch per kernel for correctness; its times are not
the 20-launch benchmark measurements. [Full GPU table](gpu_results.md),
[raw results](gpu_results.json).

## 2. New variants and failures

There are 76 attempted DL expressions, with regular and odd shapes. The organisers'
generator packaged 55 new problems: 53 pass on the GPU and two fail:

- `a_dl_transformer_stages_3x17`: `k_layer_norm_matmul_softmax_reduce_737148`
  exceeds the runner's 20-second launch limit.
- `a_dl_transformer_stages_8x32`: `k_layer_norm_matmul_softmax_reduce_8e8501`
  exceeds the same limit.

Both remain in the ordinary test suite. Their independently rendered Emmy CUDA,
compiled with NVRTC, also times out at those kernels. This suggests a shared schedule
problem but does not establish its cause. [Control results](transformer_reference.json).

The other 21 attempts fail tracing, lowering, materialization or package validation.
They do not count as GPU passes. Every variant and error is listed in
[DL_COVERAGE.md](DL_COVERAGE.md), with exact generator logs linked there.

New passing coverage includes embedding gather, both RoPE halves, affine norms,
softmax and decomposed log-softmax, dense soft-target cross-entropy, SwiGLU, GELU
and deeper MLPs, scaled/bias attention, KV single-query attention, odd-size causal
attention, prepared convolution/patch matmuls, pooling reductions, flattened global
average pooling, inference BatchNorm, channel affine transforms, mel-style logs,
cumsum, residual normalization, L2 normalization, cosine similarity and masking.

## 3. Emmy limitations

Native log-softmax, cross-entropy, avg/max pool2d and argmax do not lower through these
probes. Multi-axis global pooling and the target-gather loss encounter unbound
coordinate names. The fully fused transformer fails materialization. Constant causal
masks are packaged with null data; one dynamic scan schedule lacks a per-kernel dump.

Gather and scan already exist and pass tests. Tensor-core and bf16/fp8 constructs
also appear in the installed source; they are not falsely listed as absent IR types.
The [one-page wishlist](IR_WISHLIST.md) distinguishes observed gaps from untested proposals.

## 4. EmmyPTX versus Emmy + NVRTC

Actual nvcc was not measured. Both paths use the same scheduled IR and launch contract,
targeting sm_89/compute_89. Only CUDA parameter declarations are reordered to match the
metadata ABI; the rendered body is unchanged. Fast math is disabled, as is implicit
CUDA multiply-add contraction; EmmyPTX still fuses explicit accumulation products.

All 29 evaluated size-sweep workloads pass both implementations. The two 2^24-element
pointwise cases are skipped because the runner can return at most 2^22 output values.

Compile medians across the 29 validated workload medians:

| Measurement | EmmyPTX | Emmy + NVRTC |
|---|---:|---:|
| Fresh process, IR to PTX | 737.494 ms | 889.769 ms |
| Warm IR compilation | 3.971 ms | not measured separately |
| Preloaded lowering / NVRTC compile call | 2.095 ms | 39.464 ms |

The last row compares different compilation stages and excludes startup/rendering.
Each workload has one warm-up and seven measured compilation repetitions. GPU results
use three alternating paired rounds of 3 warm-ups and 20 launches, without client
response caching. Clocks are not locked.

Compilation was measured during the broader generation/validation work, so host load
can affect these times. Treat them as indicative measurements, not isolated CPU benchmarks.

Examples of GPU execution medians:

| Workload | EmmyPTX | Emmy + NVRTC |
|---|---:|---:|
| Matmul 1024x1024 | 78.464 us | 100.768 us |
| Attention, sequence 64 and head 64 | 84.640 us | 152.160 us |
| Attention, sequence 256 and head 64 | 952.880 us | 844.208 us |

These are sums of kernel medians, excluding transfers and model startup. No speed
claim applies universally; the tables include wins, losses and near-equal results.
No cause for a speed difference is asserted without profiling.

Artifacts, all with white backgrounds:

- [Execution sweep PNG](bench/size_sweep.png), [SVG](bench/size_sweep.svg).
- [Compile times PNG](bench/compile_times.png), [SVG](bench/compile_times.svg).
- [Full comparison](bench/RESULTS_nvcc.md), [raw samples and versions](bench/raw_results.json).

## 5. End-to-end demo

The MLP demo ran on the local NVIDIA RTX 3050 Laptop GPU, without fallback. It defines
a PyTorch module with widths 32, 64 and 10, traces it through Emmy, compiles both
generated kernels using compile.py, and chains the hidden-layer buffer on the GPU.
Hidden activations and logits match PyTorch at rtol=1e-3, atol=1e-4.

The recorded maximum absolute error is 1.52588e-5; maximum relative error is about
3.75868e-6, with denominator floor 1e-8. The two local kernel medians are 8.192 and
7.168 us. They are separate kernel measurements, not total inference latency.
[Recorded local result](demo_result.json).

Only the demo's in-memory PTX target changes to sm_86 before loading. The compiler
continues to emit sm_89. The demo also provides an explicit remote mode and an automatic
fallback for local loading/execution errors; numerical mismatches remain failures.

## 6. Compiler changes, tooling and commits

The only compiler change is leaf integer CastExpr support in index.py. It delegates
to the existing typed index resolver, including its i64-to-s32 conversion. It is
general, independent of problem names/shapes, and has positive and rejection tests.
Compound casts and unsupported cast types remain errors.

Generation calls the organisers' unchanged trace/package functions. A Windows CLI
adapter handles the POSIX fcntl import and UTF-8 dumps. Package validation records
null constants and shape mismatches. Launch tooling orders intermediate dependencies,
retains failures, checks missing/truncated outputs and avoids narrowing output values.

The NVRTC comparison and direct renderer adapter stay inside bench/. The generator
uses Emmy's standard lowering as required. The submission compiler imports none of
the comparison tools. Tests check that boundary. No CUDA toolkit was installed.

Commits:

- `e62a6ca`: Support integer casts on gather indices.
- `0a410cd`: Add generated DL probes and validate launch contracts.
- `2a46f0b`: Add isolated NVRTC comparison and local MLP demo.
- The final documentation commit contains this report, results, plots and the revised
  4-minute-40-second video script. Its hash is included in the completion message.

All commit messages have plain subjects with no attribution footers. Nothing was pushed.

## 7. Remaining work and reproduction

The full transformer does not run successfully, and the native frontend/package
limitations above remain. No risky compiler workaround was introduced. Convolution
tests start with prepared windows, so they do not claim full image-to-im2col coverage.
The small model uses synthetic data and makes no accuracy claim.

The tested environment is Python 3.12.11, emmy-ml 0.3.11, CPU torch 2.14.1+cpu,
NumPy 2.5.4, SciPy 1.18.1, CuPy 14.2.0, NVRTC wheel 12.8.93, matplotlib 3.11.2,
pytest 9.1.1 and Ruff 0.17.3. Local driver: 572.83. Organiser checkout:
`ec8cde1943f88efea3827a95bda6085553d2d407`.

```bash
python -m pytest tests
python -m ruff check .
python tools/run_all.py --no-cache --json docs/gpu_results.json --markdown docs/gpu_results.md
python bench/demo_e2e.py
python bench/compare_nvcc.py --resume --reuse-generated
```

The whole GPU suite intentionally exits nonzero while the two transformer failures
remain. Generator probes also exit nonzero for unsupported attempts. Cached reruns
are available for review, but the final full correctness run used fresh GPU requests.
