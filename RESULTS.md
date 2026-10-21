# Results

Measured on the hackathon runner (RTX 5090, sm_120, PTX targeting sm_89) with
`python tools/run_all.py --timing --markdown ...`: every kernel is compiled by
`compile.py`, assembled by the runner's `ptxas`, launched with its `meta.json`
geometry, and compared with `expected.npz` using the problem's rtol / atol. Timings are the
runner's median over 20 launches after 3 warm-ups (indicative only: the runner does not lock
clocks). Compile time is about 0.4 s per kernel, most of it importing emmy.

The open set is the 15 problems of the hackathon repo. The generated variants are built from
`tests/variants.py` by the organizers' own generator (see README).

## Pass table

| category | open set | generated variants |
|---|---|---|
| pointwise | 4/4 | 38/38 |
| reductions | 4/4 | 27/27 |
| matmul | 2/2 | 13/13 |
| fused | 3/3 | 14/14 |
| attention | 2/2 | 8/8 |

| suite | problem | result | median us | registers |
|---|---|---|---|---|
| open | a01_attention | PASS | 84.6 | 128 |
| open | a02_attention_128 | PASS | 41.5 | 80 |
| open | f01_rmsnorm | PASS | 4.7 | 22 |
| open | f02_row_softmax | PASS | 3.1 | 22 |
| open | f03_l2norm | PASS | 3.2 | 22 |
| open | m01_matmul_512 | PASS | 15.0 | 40 |
| open | m02_matmul_rect | PASS | 4.9 | 35 |
| open | p00_scalar_add | PASS | 3.1 | 12 |
| open | p01_vector_add | PASS | 3.2 | 12 |
| open | p02_scale_shift | PASS | 3.2 | 21 |
| open | p03_relu_mul | PASS | 3.1 | 22 |
| open | r01_array_sum | PASS | 3.2 | 18 |
| open | r02_big_sum | PASS | 21.1 | 18 |
| open | r03_mean | PASS | 12.9 | 18 |
| open | r04_row_sum | PASS | 3.1 | 22 |
| generated | a_attn_256 | PASS | 3122.5 | 128 |
| generated | a_attn_512 | PASS | 191.1 | 224 |
| generated | a_attn_64x32 | PASS | 38.4 | 128 |
| generated | a_attn_batched | PASS | 39.5 | 170 |
| generated | a_attn_d128 | PASS | 826.4 | 40 |
| generated | a_attn_noscale | PASS | 25.2 | 40 |
| generated | a_attn_odd | PASS | 109.2 | 80 |
| generated | a_matmul_softmax | PASS | 957.1 | 248 |
| generated | f_center | PASS | 4.8 | 18 |
| generated | f_l2norm_odd | PASS | 3.2 | 22 |
| generated | f_layernorm | PASS | 4.8 | 40 |
| generated | f_layernorm_affine | PASS | 6.7 | 40 |
| generated | f_rmsnorm_512 | PASS | 4.7 | 22 |
| generated | f_rmsnorm_block256 | PASS | 4.7 | 22 |
| generated | f_rmsnorm_odd | PASS | 4.7 | 22 |
| generated | f_softmax_col | PASS | 8.9 | 24 |
| generated | f_softmax_f16 | PASS | 3.2 | 22 |
| generated | f_softmax_odd | PASS | 4.7 | 22 |
| generated | f_softmax_wide | PASS | 4.7 | 22 |
| generated | f_sub_max | PASS | 3.1 | 20 |
| generated | f_sumsq_normalize | PASS | 3.1 | 22 |
| generated | f_var_row | PASS | 4.7 | 22 |
| generated | m_batched | PASS | 5.0 | 48 |
| generated | m_bias | PASS | 5.7 | 64 |
| generated | m_large | PASS | 49.8 | 72 |
| generated | m_matmul_rowsum | PASS | 281.7 | 40 |
| generated | m_mlp | PASS | 29.3 | 80 |
| generated | m_odd | PASS | 6.8 | 36 |
| generated | m_rect | PASS | 5.0 | 60 |
| generated | m_relu | PASS | 5.9 | 60 |
| generated | m_small | PASS | 4.5 | 40 |
| generated | m_square_256 | PASS | 12.7 | 36 |
| generated | m_tall | PASS | 4.9 | 40 |
| generated | m_vecmat | PASS | 3.5 | 24 |
| generated | m_wide_k | PASS | 4.8 | 48 |
| generated | p_4d_bcast | PASS | 3.2 | 22 |
| generated | p_add_odd | PASS | 3.1 | 12 |
| generated | p_add_tiny | PASS | 3.1 | 12 |
| generated | p_bcast_3d | PASS | 3.1 | 16 |
| generated | p_bcast_col | PASS | 3.1 | 18 |
| generated | p_bcast_row | PASS | 3.1 | 22 |
| generated | p_big_odd | PASS | 3.2 | 12 |
| generated | p_chain | PASS | 3.2 | 23 |
| generated | p_clamp | PASS | 3.2 | 15 |
| generated | p_div | PASS | 3.2 | 24 |
| generated | p_exp | PASS | 3.1 | 18 |
| generated | p_f16_exp | PASS | 3.1 | 12 |
| generated | p_f16_mul | PASS | 3.2 | 16 |
| generated | p_f16_sigmoid | PASS | 3.1 | 15 |
| generated | p_f16_to_f32 | PASS | 3.2 | 13 |
| generated | p_f64_add | PASS | 3.1 | 22 |
| generated | p_gelu | PASS | 3.3 | 22 |
| generated | p_gelu_tanh | PASS | 3.4 | 20 |
| generated | p_int32 | PASS | 3.2 | 18 |
| generated | p_int64 | PASS | 3.2 | 26 |
| generated | p_int_float | PASS | 3.1 | 16 |
| generated | p_log_abs | PASS | 3.2 | 15 |
| generated | p_mask_float | PASS | 3.3 | 16 |
| generated | p_max_min | PASS | 3.1 | 16 |
| generated | p_mul_big | PASS | 3.2 | 22 |
| generated | p_relu_neg_abs | PASS | 3.1 | 12 |
| generated | p_rsqrt | PASS | 3.2 | 14 |
| generated | p_scale_shift_odd | PASS | 3.3 | 20 |
| generated | p_sigmoid | PASS | 3.1 | 16 |
| generated | p_silu | PASS | 3.2 | 17 |
| generated | p_sin_cos | PASS | 3.2 | 13 |
| generated | p_sin_wide | PASS | 3.2 | 14 |
| generated | p_sqrt | PASS | 3.2 | 14 |
| generated | p_square_pow | PASS | 3.2 | 16 |
| generated | p_sub_2d | PASS | 3.2 | 30 |
| generated | p_tanh | PASS | 3.1 | 18 |
| generated | p_transpose | PASS | 3.1 | 20 |
| generated | p_where | PASS | 3.3 | 18 |
| generated | r_3d_last | PASS | 3.2 | 15 |
| generated | r_abs_sum | PASS | 3.1 | 22 |
| generated | r_block1024_max | PASS | 90.6 | 40 |
| generated | r_block16 | PASS | 3.1 | 18 |
| generated | r_block256 | PASS | 3.2 | 22 |
| generated | r_block32 | PASS | 4.7 | 18 |
| generated | r_block512 | PASS | 4.7 | 22 |
| generated | r_block64_max | PASS | 3.2 | 20 |
| generated | r_col_mean_split | PASS | 5.3 | 22 |
| generated | r_col_sum | PASS | 3.2 | 22 |
| generated | r_cumsum | PASS | 6.7 | 40 |
| generated | r_f16_sum | PASS | 4.7 | 20 |
| generated | r_int_row_max | PASS | 3.8 | 22 |
| generated | r_int_sum | PASS | 6.8 | 24 |
| generated | r_max | PASS | 4.7 | 18 |
| generated | r_max_unpinned | PASS | 31.3 | 18 |
| generated | r_mean_odd | PASS | 4.8 | 18 |
| generated | r_mid_axis | PASS | 3.2 | 22 |
| generated | r_row_max | PASS | 3.1 | 20 |
| generated | r_row_mean | PASS | 3.3 | 22 |
| generated | r_row_sum_odd | PASS | 3.1 | 22 |
| generated | r_row_sum_unpinned | PASS | 3.1 | 22 |
| generated | r_sum_big | PASS | 60.0 | 18 |
| generated | r_sum_odd | PASS | 3.3 | 18 |
| generated | r_sum_split | PASS | 5.4 | 20 |
| generated | r_sum_then_mean | PASS | 6.7 | 40 |
| generated | r_sumsq | PASS | 4.7 | 18 |

## Optimization history

| change | m01_matmul_512 | m_rect | m_small |
|---|---|---|---|
| first working version | 29.3 us | 7.1 us | 5.4 us |
| unroll `unroll=True` loops + affine index normalization | 15.0 us | 5.1 us | 3.4 us |

Merging neighbouring loads into vector loads (A/B measured back to back, same session):
m_large 55.8 to 49.8 us, m_bias 6.7 to 5.3 us, others within noise. Merging shared-memory
loads inside loops that stay rolled made m_rect slower (5.2 to 7.0 us), so the compiler only
merges shared loads inside loops it unrolls itself.

## A bug found in emmy 0.3.11 while testing

`x = torch.randn(64, 256); ((x - x.mean(-1, keepdim=True)) ** 2).mean(-1)` (row variance),
when left to emmy's own schedule, is split across two blocks per row with an atomic add. In
the generated IR the first loop (the row mean) only visits `x[a0, 0..128]` and has no
`_ksplit` offset, yet divides by 256, so the IR itself computes the wrong value. Our kernel
matches the IR's literal semantics to 1e-7 (checked against a numpy model of the IR) and
therefore differs from the true variance by 1e-2. Since the organizers check that emmy's own
kernel passes every hidden problem, such an IR cannot be in the hidden set; the variant in
our suite pins the cooperative schedule instead, which emmy lowers correctly.
