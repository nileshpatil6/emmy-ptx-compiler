# Results

## DL extension

The baseline was 193/193 GPU problems and 199 pytest checks. The extension probes 76 DL
expressions and packages 55 new problems. The expanded suite has 248 problems; 246 pass
and two staged transformer problems time out. Unsupported tracing/packaging attempts
are separate from these totals. All original 193 problems still pass.

| category | before | after | open | generated | synthetic |
|---|---:|---:|---:|---:|---:|
| pointwise | 65/65 | 77/77 | 4/4 | 69/69 | 4/4 |
| reductions | 44/44 | 52/52 | 4/4 | 46/46 | 2/2 |
| matmul | 31/31 | 43/43 | 2/2 | 41/41 | - |
| fused | 34/34 | 50/50 | 3/3 | 47/47 | - |
| attention | 19/19 | 24/26 | 2/2 | 22/24 | - |

The fresh full GPU run uses `python tools/run_all.py --no-cache --json docs/gpu_results.json
--markdown docs/gpu_results.md`. These are correctness runs with one launch, not the
20-launch timing measurements below. See the [full GPU table](docs/gpu_results.md),
[DL coverage and failures](docs/DL_COVERAGE.md) and [operator wishlist](docs/IR_WISHLIST.md).

The only compiler change accepts a leaf `CastExpr('int', ...)` in an index expression,
using the existing typed resolver and integer-width conversion. It enables embedding
gathers without changing the compiler target or using another compiler. Compound and
non-integer casts still fail explicitly. Launch tooling now sorts intermediate dependencies
and reports rejected requests without dropping the rest of the suite.

The [Emmy + NVRTC comparison](docs/bench/RESULTS_nvcc.md) contains compile medians, execution
size sweeps and raw samples. Actual nvcc was not measured. Both implementations use the
same scheduled IR and launch contract. The 2^24-element pointwise cases are skipped because
the runner cannot return all output values for validation.

The [MLP demo](docs/demo_result.json) ran locally on the RTX 3050 Laptop GPU in two kernels,
including a chained hidden-layer buffer. It matched PyTorch with maximum absolute error
1.52588e-5 and maximum relative error about 3.76e-6 in the recorded run. This is an inference
correctness demo on synthetic data; no classification accuracy claim is made.

## Earlier measurements: the 193-problem baseline

Measured on the hackathon runner (RTX 5090, sm_120, PTX targeting sm_89) with
`python tools/run_all.py --timing --markdown ...`: every kernel is compiled by `compile.py`,
assembled by the runner's `ptxas`, launched with its `meta.json` geometry, and compared with
`expected.npz` using the problem's rtol / atol. Timings are the runner's median over 20
launches after 3 warm-ups (indicative only: clocks are not locked, and kernels under about
5 us are at the launch-overhead floor). Compile time is about 0.4 s per kernel, most of it
importing emmy.

Three suites:

* **open**: the 15 problems of the hackathon repo.
* **generated**: 172 problems built from `tests/variants.py` by the organizers' own generator
  (see README), our stand-in for the hidden set.
* **synthetic**: 6 hand-built IRs (`tests/synthetic.py`) for statement forms emmy rarely
  emits: Cond with else, Select chains, integer Let, eight-wide f16 vectors, producer (aux)
  threads with warp and named barriers, 4- and 8-byte cp.async, a segmented TreeHalve, and a
  StridedLoop with an explicit end feeding an int64 warp shuffle.

## Pass table

| category | open set | generated variants | synthetic IR |
|---|---|---|---|
| pointwise | 4/4 | 57/57 | 4/4 |
| reductions | 4/4 | 38/38 | 2/2 |
| matmul | 2/2 | 29/29 | - |
| fused | 3/3 | 31/31 | - |
| attention | 2/2 | 17/17 | - |

| suite | problem | result | median us | registers |
|---|---|---|---|---|
| open | a01_attention | PASS | 85.2 | 128 |
| open | a02_attention_128 | PASS | 37.4 | 80 |
| open | f01_rmsnorm | PASS | 4.8 | 40 |
| open | f02_row_softmax | PASS | 3.3 | 38 |
| open | f03_l2norm | PASS | 3.0 | 32 |
| open | m01_matmul_512 | PASS | 15.0 | 40 |
| open | m02_matmul_rect | PASS | 5.0 | 35 |
| open | p00_scalar_add | PASS | 3.1 | 12 |
| open | p01_vector_add | PASS | 3.1 | 12 |
| open | p02_scale_shift | PASS | 3.2 | 21 |
| open | p03_relu_mul | PASS | 3.0 | 22 |
| open | r01_array_sum | PASS | 3.1 | 26 |
| open | r02_big_sum | PASS | 10.8 | 36 |
| open | r03_mean | PASS | 6.8 | 36 |
| open | r04_row_sum | PASS | 3.0 | 22 |
| generated | a_attn_1024 | PASS | 489.9 | 218 |
| generated | a_attn_256 | PASS | 957.9 | 128 |
| generated | a_attn_512 | PASS | 191.1 | 222 |
| generated | a_attn_64x32 | PASS | 19.1 | 130 |
| generated | a_attn_batched | PASS | 25.1 | 255 |
| generated | a_attn_bias | PASS | 8.8 | 56 |
| generated | a_attn_d128 | PASS | 903.6 | 64 |
| generated | a_attn_dv_diff | PASS | 621.0 | 128 |
| generated | a_attn_heads4 | PASS | 74.3 | 220 |
| generated | a_attn_heads8_d16 | PASS | 27.3 | 128 |
| generated | a_attn_noscale | PASS | 25.2 | 40 |
| generated | a_attn_odd | PASS | 100.9 | 78 |
| generated | a_attn_q16_k512 | PASS | 390.7 | 100 |
| generated | a_attn_seq48 | PASS | 23.1 | 64 |
| generated | a_matmul_softmax | PASS | 1010.5 | 248 |
| generated | a_scores_softmax | PASS | 207.4 | 80 |
| generated | a_softmax_matmul | PASS | 35.4 | 77 |
| generated | f_bias_softmax | PASS | 3.3 | 38 |
| generated | f_center | PASS | 4.7 | 40 |
| generated | f_l2norm_odd | PASS | 3.1 | 38 |
| generated | f_layernorm | PASS | 4.7 | 40 |
| generated | f_layernorm_768 | PASS | 8.8 | 40 |
| generated | f_layernorm_affine | PASS | 4.8 | 40 |
| generated | f_layernorm_f16 | PASS | 6.6 | 40 |
| generated | f_max_norm | PASS | 3.1 | 20 |
| generated | f_mean_center_scale | PASS | 3.1 | 22 |
| generated | f_norm_dim0 | PASS | 15.0 | 22 |
| generated | f_rmsnorm_4096 | PASS | 6.8 | 40 |
| generated | f_rmsnorm_512 | PASS | 3.4 | 38 |
| generated | f_rmsnorm_block256 | PASS | 4.7 | 40 |
| generated | f_rmsnorm_f16 | PASS | 3.3 | 36 |
| generated | f_rmsnorm_odd | PASS | 4.7 | 22 |
| generated | f_rmsnorm_small | PASS | 4.6 | 22 |
| generated | f_sigmoid_mean | PASS | 3.3 | 20 |
| generated | f_silu_gate | PASS | 4.7 | 22 |
| generated | f_softmax_1x4096 | PASS | 5.1 | 40 |
| generated | f_softmax_3d | PASS | 4.7 | 36 |
| generated | f_softmax_col | PASS | 14.9 | 38 |
| generated | f_softmax_dim0_odd | PASS | 8.9 | 38 |
| generated | f_softmax_f16 | PASS | 4.7 | 34 |
| generated | f_softmax_mid | PASS | 12.9 | 39 |
| generated | f_softmax_odd | PASS | 4.7 | 38 |
| generated | f_softmax_scaled | PASS | 3.2 | 39 |
| generated | f_softmax_wide | PASS | 4.7 | 40 |
| generated | f_std_row | PASS | 3.2 | 22 |
| generated | f_sub_max | PASS | 3.0 | 20 |
| generated | f_sumsq_normalize | PASS | 3.0 | 22 |
| generated | f_var_row | PASS | 4.3 | 22 |
| generated | m_1000x300x200 | PASS | 39.6 | 56 |
| generated | m_2048x64 | PASS | 11.2 | 44 |
| generated | m_768 | PASS | 60.0 | 64 |
| generated | m_batched | PASS | 5.1 | 48 |
| generated | m_bias | PASS | 4.8 | 64 |
| generated | m_bmm8 | PASS | 3.2 | 40 |
| generated | m_bmm_odd | PASS | 9.1 | 64 |
| generated | m_gelu_epi | PASS | 7.8 | 39 |
| generated | m_k3 | PASS | 3.1 | 24 |
| generated | m_k4096 | PASS | 9.2 | 114 |
| generated | m_large | PASS | 27.3 | 114 |
| generated | m_matmul_rowsum | PASS | 279.7 | 48 |
| generated | m_mlp | PASS | 29.3 | 80 |
| generated | m_odd | PASS | 6.8 | 36 |
| generated | m_odd_all | PASS | 6.8 | 34 |
| generated | m_raster | PASS | 271.0 | 40 |
| generated | m_rect | PASS | 5.1 | 64 |
| generated | m_relu | PASS | 4.7 | 64 |
| generated | m_residual | PASS | 8.8 | 41 |
| generated | m_scaled | PASS | 3.3 | 40 |
| generated | m_silu_epi | PASS | 4.7 | 62 |
| generated | m_small | PASS | 3.4 | 40 |
| generated | m_square_256 | PASS | 6.8 | 36 |
| generated | m_stage_sync | PASS | 8.8 | 124 |
| generated | m_tall | PASS | 4.9 | 40 |
| generated | m_tanh_epi | PASS | 4.7 | 37 |
| generated | m_vecmat | PASS | 3.5 | 24 |
| generated | m_wide_k | PASS | 4.8 | 48 |
| generated | m_xt | PASS | 5.1 | 40 |
| generated | p_4d_bcast | PASS | 3.0 | 22 |
| generated | p_add3 | PASS | 3.1 | 26 |
| generated | p_add_odd | PASS | 3.2 | 12 |
| generated | p_add_tiny | PASS | 3.1 | 12 |
| generated | p_bcast_3d | PASS | 3.1 | 16 |
| generated | p_bcast_both | PASS | 3.0 | 18 |
| generated | p_bcast_col | PASS | 3.1 | 18 |
| generated | p_bcast_row | PASS | 3.1 | 22 |
| generated | p_bcast_scalar_tensor | PASS | 3.1 | 22 |
| generated | p_big_odd | PASS | 3.8 | 12 |
| generated | p_chain | PASS | 3.1 | 24 |
| generated | p_clamp | PASS | 3.1 | 15 |
| generated | p_div | PASS | 3.0 | 24 |
| generated | p_div_scalar | PASS | 3.1 | 17 |
| generated | p_exp | PASS | 3.0 | 21 |
| generated | p_exp_large | PASS | 3.1 | 21 |
| generated | p_f16_chain | PASS | 3.1 | 24 |
| generated | p_f16_exp | PASS | 3.0 | 16 |
| generated | p_f16_mul | PASS | 3.1 | 16 |
| generated | p_f16_sigmoid | PASS | 3.0 | 15 |
| generated | p_f16_to_f32 | PASS | 3.1 | 13 |
| generated | p_f64_add | PASS | 3.1 | 22 |
| generated | p_f64_exp | PASS | 3.0 | 15 |
| generated | p_gelu | PASS | 3.3 | 22 |
| generated | p_gelu_tanh | PASS | 3.3 | 20 |
| generated | p_hardsigmoid_like | PASS | 3.3 | 20 |
| generated | p_int32 | PASS | 3.2 | 18 |
| generated | p_int64 | PASS | 3.2 | 26 |
| generated | p_int_float | PASS | 3.1 | 16 |
| generated | p_log_abs | PASS | 3.6 | 15 |
| generated | p_log_small | PASS | 3.0 | 15 |
| generated | p_long_chain | PASS | 4.7 | 24 |
| generated | p_mask_float | PASS | 3.1 | 16 |
| generated | p_max_min | PASS | 3.1 | 16 |
| generated | p_minimum_scalar | PASS | 3.1 | 18 |
| generated | p_mish | PASS | 3.1 | 23 |
| generated | p_mul_big | PASS | 3.1 | 22 |
| generated | p_pow_frac | PASS | 3.1 | 18 |
| generated | p_reciprocal | PASS | 3.1 | 18 |
| generated | p_relu_neg_abs | PASS | 3.0 | 12 |
| generated | p_rsqrt | PASS | 3.1 | 14 |
| generated | p_rsqrt_wide | PASS | 3.1 | 17 |
| generated | p_scale_shift_odd | PASS | 3.1 | 20 |
| generated | p_sigmoid | PASS | 3.0 | 16 |
| generated | p_sigmoid_f16_2d | PASS | 3.0 | 17 |
| generated | p_silu | PASS | 3.1 | 18 |
| generated | p_sin_cos | PASS | 3.0 | 19 |
| generated | p_sin_wide | PASS | 3.1 | 16 |
| generated | p_softplus | PASS | 3.1 | 26 |
| generated | p_sqrt | PASS | 3.1 | 14 |
| generated | p_square_pow | PASS | 3.1 | 16 |
| generated | p_sub_2d | PASS | 3.0 | 30 |
| generated | p_tanh | PASS | 3.0 | 20 |
| generated | p_tanh_small | PASS | 3.1 | 20 |
| generated | p_transpose | PASS | 3.1 | 20 |
| generated | p_where | PASS | 3.3 | 18 |
| generated | p_where_scalar | PASS | 3.1 | 19 |
| generated | r_3d_last | PASS | 3.0 | 15 |
| generated | r_abs_sum | PASS | 3.0 | 22 |
| generated | r_amax_dim0 | PASS | 4.7 | 20 |
| generated | r_amax_rows_odd | PASS | 3.0 | 20 |
| generated | r_block1024_max | PASS | 90.7 | 40 |
| generated | r_block16 | PASS | 3.1 | 24 |
| generated | r_block256 | PASS | 3.2 | 22 |
| generated | r_block32 | PASS | 4.7 | 44 |
| generated | r_block512 | PASS | 4.7 | 28 |
| generated | r_block64_max | PASS | 3.6 | 20 |
| generated | r_col_mean_split | PASS | 5.4 | 22 |
| generated | r_col_sum | PASS | 3.2 | 22 |
| generated | r_cumsum | PASS | 4.7 | 40 |
| generated | r_exp_sum | PASS | 3.4 | 25 |
| generated | r_f16_sum | PASS | 3.0 | 36 |
| generated | r_int_row_max | PASS | 3.0 | 22 |
| generated | r_int_sum | PASS | 6.7 | 38 |
| generated | r_max | PASS | 4.7 | 34 |
| generated | r_max_unpinned | PASS | 15.0 | 34 |
| generated | r_mean_abs | PASS | 4.7 | 36 |
| generated | r_mean_last_3d | PASS | 4.7 | 15 |
| generated | r_mean_odd | PASS | 4.8 | 36 |
| generated | r_mid_axis | PASS | 3.1 | 22 |
| generated | r_norm2 | PASS | 3.0 | 22 |
| generated | r_row_max | PASS | 3.1 | 20 |
| generated | r_row_mean | PASS | 3.1 | 28 |
| generated | r_row_sum_odd | PASS | 3.0 | 22 |
| generated | r_row_sum_unpinned | PASS | 3.0 | 22 |
| generated | r_sum_1e6 | PASS | 6.5 | 38 |
| generated | r_sum_axis0_big | PASS | 3.6 | 22 |
| generated | r_sum_big | PASS | 25.2 | 36 |
| generated | r_sum_f16_rows | PASS | 3.1 | 22 |
| generated | r_sum_keepdim | PASS | 3.3 | 21 |
| generated | r_sum_mid | PASS | 3.1 | 22 |
| generated | r_sum_odd | PASS | 3.1 | 26 |
| generated | r_sum_split | PASS | 4.5 | 38 |
| generated | r_sum_then_mean | PASS | 6.8 | 40 |
| generated | r_sumsq | PASS | 3.0 | 36 |
| synthetic | p_syn_aux_barriers | PASS | 3.1 | 10 |
| synthetic | p_syn_cond_else_select | PASS | 3.1 | 14 |
| synthetic | p_syn_cp_async_4_8 | PASS | 3.1 | 18 |
| synthetic | p_syn_f16_vector8 | PASS | 3.0 | 17 |
| synthetic | r_syn_int64_shuffle_end | PASS | 4.7 | 40 |
| synthetic | r_syn_segmented_tree | PASS | 3.0 | 26 |

## Optimization history

Every optimization was measured A/B on the runner (old and new compiler back to back in one
session) over the whole suite, and kept only if no kernel got slower beyond noise, or, for
the unroll budget, if the total gain clearly outweighed one slower kernel.

| change | effect (median us, before -> after) |
|---|---|
| unroll loops the IR marks `unroll=True`, affine index normalization | m01_matmul_512 29.3 -> 15.0, m_small 5.4 -> 3.4 |
| merge neighbouring aligned loads into v2 / v4 | m_large 55.8 -> 49.8, m_bias 6.7 -> 5.3 |
| partially unroll other innermost constant-trip loops by 4 | geomean 0.90 over kernels above 15 us; a_attn_256 3125 -> 596 (with the next rows), m_large 49.8 -> 31.4 |
| strided loops in in-range blocks of up to 16 iterations | geomean 0.91 over kernels above 6 us; r02_big_sum 21.1 -> 10.9, r03_mean 12.9 -> 6.8, r_sum_big 58.0 -> 25.2 |
| strided unrolling capped at the loop's maximum trip count (no prefix under 4 trips) | geomean 0.98; f02_row_softmax 4.7 -> 3.3 |
| full-unroll budget 4096 -> 8192 statements | geomean 0.96; a_attn_64x32 39.4 -> 19.1, a_attn_batched 39.6 -> 25.2, m_large 32.1 -> 27.3, a_attn_d128 813 -> 903 (slower) |

Rejected after measuring: fully unrolling every small innermost loop (a01 1.8x faster but
a_attn_512 39% slower), unroll factors 8 and 32 (32 slowed kernels with large bodies), and
merging shared-memory loads in loops that stay rolled (m_rect 5.2 -> 7.0).

## Precision of the math functions

Measured on the GPU against float64 numpy (max relative error where |ref| > 1e-3; absolute
error otherwise). In straight-line code: exp uses Cody-Waite reduction and an exact 2^n, sin and
cos a quadrant reduction and polynomials, tanh and erf Taylor series near zero. Inside loops
(online softmax, attention) exp uses the two-instruction sequence of CUDA's `__expf`, where the
reductions' tolerances are 1e-3; this kept their speed unchanged.

| function | max relative error |
|---|---|
| exp | 1.3e-7 |
| sin, cos | 1.4e-7 |
| tanh | 2.2e-7 |
| sigmoid, silu | 2.2e-7 |
| sqrt, rsqrt | 1.2e-7 |
| softmax (loop exp) | 4.8e-7 |

## Bugs found in emmy 0.3.11 while testing

1. **Row variance under emmy's own split schedule.**
   `((x - x.mean(-1, keepdim=True)) ** 2).mean(-1)` is split across two blocks per row with an
   atomic add, but the IR's mean loop only visits `x[a0, 0..128]` (no `_ksplit` offset) and
   still divides by 256. Our kernel matches the IR's literal semantics to 1e-7 (checked against
   a numpy model of the IR), so it differs from the true variance by 1e-2. Our suite pins the
   cooperative schedule for this variant, which emmy lowers correctly.
2. **Matrix times a single column** (`(512, 256) @ (256, 1)`). The IR issues 16-byte
   `cp.async` copies from `x1` at element offsets 4j+1, 4j+2, 4j+3, which are not 16-byte
   aligned (emmy's own kernel would fault), and the copy moves four consecutive k values into
   four different columns. Our compiler proves alignment where it can and otherwise emits a
   run-time check with an element-wise fallback, so it never faults; the result still follows
   the (wrong) IR.

Neither IR can be in the hidden set, since the organizers check that emmy's own kernel passes
every problem. Two other differences from numpy follow emmy by design: integer `//` and `%`
truncate toward zero as in emmy's C, and `x / c` with a constant is emmy's `x * (1 / c)`,
one ulp from numpy's division.
