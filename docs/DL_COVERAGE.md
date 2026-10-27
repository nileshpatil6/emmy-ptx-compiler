# DL coverage

76 attempted expressions; 55 packaged problems. Regular shapes are 8x32; odd shapes are 3x17.
Fixture presence does not imply GPU correctness. Every runtime failure is retained in the normal suite.

All new computations are inside explicit PyTorch modules. Input order is the module signature; references use independent NumPy calculations.
The original organiser trace/package functions generate the IR, launch contracts and expected arrays.

| Variant | Generation / GPU outcome | Detail |
|---|---|---|
| `p_dl_embedding_8x32` | packaged; PASS | 1 kernel(s) |
| `p_dl_rope_8x32` | packaged; PASS | 2 kernel(s) |
| `f_dl_rmsnorm_8x32` | packaged; PASS | 1 kernel(s) |
| `f_dl_layernorm_8x32` | packaged; PASS | 1 kernel(s) |
| `f_dl_softmax_8x32` | packaged; PASS | 1 kernel(s) |
| `f_dl_logsoftmax_8x32` | trace failure | ValueError: unknown elementwise op name: 'log_softmax' (not in numpy or _NAME_TO_FN) |
| `f_dl_cross_entropy_8x32` | trace failure | ValueError: cannot map fallback aten.cross_entropy_loss as elementwise for output shape () |
| `f_dl_cross_entropy_dense_8x32` | packaged; PASS | 1 kernel(s) |
| `m_dl_swiglu_8x32` | packaged; PASS | 1 kernel(s) |
| `m_dl_gelu_mlp_8x32` | packaged; PASS | 1 kernel(s) |
| `a_dl_causal_8x32` | package validation failure | Generator emitted null constant data for ge; runner cannot materialize it |
| `a_dl_scaled_bias_8x32` | packaged; PASS | 1 kernel(s) |
| `a_dl_kv_query_8x32` | packaged; PASS | 1 kernel(s) |
| `a_dl_transformer_8x32` | trace failure | AssertionError: materialize: kernel 'k_layer_norm_matmul_softmax_reduce_67df7e' reads names it never binds: ['a11', 'a11__ar0', 'a11__ar1', 'a11__ar10', 'a11__ar11', 'a11__ar12', 'a11__ar13', 'a11__ar14', 'a11__ar15', 'a11__ar16', 'a11__ar17', 'a11__ar18', 'a11__ar19', 'a11__ar2', 'a11__ar20', 'a11__ar21', 'a11__ar22', 'a11__ar23', 'a11__ar24', 'a11__ar25', 'a11__ar3', 'a11__ar4', 'a11__ar5', 'a11__ar6', 'a11__ar7', 'a11__ar8', 'a11__ar9', 'a18', 'a18__ar0', 'a18__ar1', 'a18__ar10', 'a18__ar1... |
| `m_dl_conv2d_columns_8x32` | packaged; PASS | 1 kernel(s) |
| `m_dl_conv1d_windows_8x32` | packaged; PASS | 1 kernel(s) |
| `m_dl_patch_embedding_8x32` | packaged; PASS | 1 kernel(s) |
| `r_dl_avg_pool_windows_8x32` | packaged; PASS | 1 kernel(s) |
| `r_dl_max_pool_windows_8x32` | packaged; PASS | 1 kernel(s) |
| `r_dl_avg_pool2d_8x32` | trace failure | ValueError: cannot map fallback aten.avg_pool2d as elementwise for output shape (1, 3, 4, 16) |
| `r_dl_max_pool2d_8x32` | trace failure | ValueError: cannot map fallback aten.max_pool2d as elementwise for output shape (1, 3, 4, 16) |
| `p_dl_batchnorm_inference_8x32` | packaged; PASS | 1 kernel(s) |
| `p_dl_channel_affine_8x32` | packaged; PASS | 1 kernel(s) |
| `r_dl_global_avg_pool_8x32` | trace failure | ValueError: a term reads ['out_coord_2'], which no enclosing level defines |
| `p_dl_mel_log_8x32` | packaged; PASS | 1 kernel(s) |
| `r_dl_cumsum_8x32` | packaged; PASS | 1 kernel(s) |
| `r_dl_argmax_8x32` | trace failure | ValueError: cannot map fallback aten.argmax as elementwise for output shape (8,) |
| `f_dl_residual_norm_8x32` | packaged; PASS | 1 kernel(s) |
| `f_dl_l2normalize_8x32` | packaged; PASS | 1 kernel(s) |
| `f_dl_cosine_8x32` | packaged; PASS | 1 kernel(s) |
| `p_dl_clamp_mask_8x32` | packaged; PASS | 1 kernel(s) |
| `m_dl_mlp_stack_8x32` | packaged; PASS | 1 kernel(s) |
| `p_dl_embedding_3x17` | packaged; PASS | 1 kernel(s) |
| `p_dl_rope_3x17` | packaged; PASS | 2 kernel(s) |
| `f_dl_rmsnorm_3x17` | packaged; PASS | 1 kernel(s) |
| `f_dl_layernorm_3x17` | packaged; PASS | 1 kernel(s) |
| `f_dl_softmax_3x17` | packaged; PASS | 1 kernel(s) |
| `f_dl_logsoftmax_3x17` | trace failure | ValueError: unknown elementwise op name: 'log_softmax' (not in numpy or _NAME_TO_FN) |
| `f_dl_cross_entropy_3x17` | trace failure | ValueError: cannot map fallback aten.cross_entropy_loss as elementwise for output shape () |
| `f_dl_cross_entropy_dense_3x17` | packaged; PASS | 1 kernel(s) |
| `m_dl_swiglu_3x17` | packaged; PASS | 1 kernel(s) |
| `m_dl_gelu_mlp_3x17` | packaged; PASS | 1 kernel(s) |
| `a_dl_causal_3x17` | package validation failure | Generator emitted null constant data for ge; runner cannot materialize it |
| `a_dl_scaled_bias_3x17` | packaged; PASS | 1 kernel(s) |
| `a_dl_kv_query_3x17` | packaged; PASS | 1 kernel(s) |
| `a_dl_transformer_3x17` | trace failure | AssertionError: materialize: kernel 'k_layer_norm_matmul_softmax_reduce_c8b35c' reads names it never binds: ['a11', 'a11__ar0', 'a11__ar1', 'a18', 'a18__ar0', 'a18__ar1', 'a25', 'a33', 'a33__ar0', 'a33__ar1'] |
| `m_dl_conv2d_columns_3x17` | packaged; PASS | 1 kernel(s) |
| `m_dl_conv1d_windows_3x17` | packaged; PASS | 1 kernel(s) |
| `m_dl_patch_embedding_3x17` | packaged; PASS | 1 kernel(s) |
| `r_dl_avg_pool_windows_3x17` | packaged; PASS | 1 kernel(s) |
| `r_dl_max_pool_windows_3x17` | packaged; PASS | 1 kernel(s) |
| `r_dl_avg_pool2d_3x17` | trace failure | ValueError: cannot map fallback aten.avg_pool2d as elementwise for output shape (1, 3, 1, 8) |
| `r_dl_max_pool2d_3x17` | trace failure | ValueError: cannot map fallback aten.max_pool2d as elementwise for output shape (1, 3, 1, 8) |
| `p_dl_batchnorm_inference_3x17` | packaged; PASS | 1 kernel(s) |
| `p_dl_channel_affine_3x17` | packaged; PASS | 1 kernel(s) |
| `r_dl_global_avg_pool_3x17` | trace failure | ValueError: a term reads ['out_coord_2'], which no enclosing level defines |
| `p_dl_mel_log_3x17` | packaged; PASS | 1 kernel(s) |
| `r_dl_cumsum_3x17` | packaged; PASS | 1 kernel(s) |
| `r_dl_argmax_3x17` | trace failure | ValueError: cannot map fallback aten.argmax as elementwise for output shape (3,) |
| `f_dl_residual_norm_3x17` | packaged; PASS | 1 kernel(s) |
| `f_dl_l2normalize_3x17` | packaged; PASS | 1 kernel(s) |
| `f_dl_cosine_3x17` | packaged; PASS | 1 kernel(s) |
| `p_dl_clamp_mask_3x17` | packaged; PASS | 1 kernel(s) |
| `m_dl_mlp_stack_3x17` | packaged; PASS | 1 kernel(s) |
| `f_dl_logsoftmax_decomposed_8x32` | packaged; PASS | 1 kernel(s) |
| `a_dl_transformer_split_8x32` | trace failure | AssertionError: materialize: kernel 'k_layer_norm_matmul_softmax_reduce_67df7e' reads names it never binds: ['a11', 'a11__ar0', 'a18', 'a18__ar0', 'a25', 'a33', 'a33__ar0'] |
| `f_dl_logsoftmax_decomposed_3x17` | packaged; PASS | 1 kernel(s) |
| `a_dl_transformer_split_3x17` | trace failure | AssertionError: materialize: kernel 'k_layer_norm_matmul_softmax_reduce_c8b35c' reads names it never binds: ['a11', 'a11__ar0', 'a11__ar1', 'a11__ar2', 'a11__ar3', 'a11__ar4', 'a11__ar5', 'a18', 'a18__ar0', 'a18__ar1', 'a18__ar2', 'a18__ar3', 'a18__ar4', 'a18__ar5', 'a25', 'a33', 'a33__ar0', 'a33__ar1', 'a33__ar2', 'a33__ar3', 'a33__ar4', 'a33__ar5'] |
| `a_dl_transformer_stages_8x32` | packaged; FAIL | k_layer_norm_matmul_softmax_reduce_8e8501: launch failed (run): the launch did not finish within 20 s and was stopped. Look for a loop that never exits or a barrier that not every thread reaches. |
| `r_dl_global_avg_pool_flatten_8x32` | packaged; PASS | 1 kernel(s) |
| `a_dl_causal_dynamic_8x32` | package failure | no per-kernel IR dump for k_cumsum_reduce__zp256 (node cumsum) |
| `f_dl_cross_entropy_gather_8x32` | trace failure | ValueError: a term reads ['out_coord_0'], which no enclosing level defines |
| `a_dl_transformer_stages_3x17` | packaged; FAIL | k_layer_norm_matmul_softmax_reduce_737148: launch failed (run): the launch did not finish within 20 s and was stopped. Look for a loop that never exits or a barrier that not every thread reaches. |
| `r_dl_global_avg_pool_flatten_3x17` | packaged; PASS | 1 kernel(s) |
| `a_dl_causal_dynamic_3x17` | packaged; PASS | 3 kernel(s) |
| `f_dl_cross_entropy_gather_3x17` | trace failure | ValueError: a term reads ['out_coord_0'], which no enclosing level defines |

Packaged: 55/76. Generation failures: 21.

## Interpretation

- Embedding tests bounded integer indices derived from seeded float inputs; gather remains inside the traced graph.
- RoPE returns both rotated halves. Angle generation is outside this primitive's contract.
- Dense cross-entropy uses soft targets. Native categorical cross-entropy and the explicit target-gather form fail upstream.
- Convolution, patch embedding and pooling-window cases start with prepared windows/patches. They do not test im2col or image padding.
- BatchNorm tests inference arithmetic, with positive variance constructed by abs. There is no training-state update.
- Cumsum passes; argmax does not trace. Neither is silently replaced with a different operation.
- Native causal-mask packages contain null tensor constants. The odd dynamic-mask decomposition runs; the regular one has a missing dump.
- Full transformer probes retain failures. Staged forms expose norm, scores, probabilities, residual, second norm, MLP activation and final output.
- The organisers' generator can silently drop returned intermediates during fusion. The retained stage interface and references use the actual seven observable outputs, and package shapes are validated.
- A transient interrupted package for the odd MLP stack was regenerated successfully; the retry is recorded separately.

## Exact generation records

- [dl_causal_generation.json](dl_causal_generation.json)
- [dl_cross_entropy_generation.json](dl_cross_entropy_generation.json)
- [dl_generation.json](dl_generation.json)
- [dl_logsoftmax_generation.json](dl_logsoftmax_generation.json)
- [dl_mlp_retry_generation.json](dl_mlp_retry_generation.json)
- [dl_pool_generation.json](dl_pool_generation.json)
- [dl_split_generation.json](dl_split_generation.json)
- [dl_stages_generation.json](dl_stages_generation.json)

The generation command exits nonzero while unsupported probes remain. These failures are separate from GPU pass totals.
Invalid constant packages are retained in the ignored `.cache/rejected/` folder, not counted as runnable fixtures.
The full GPU run is in [gpu_results.md](gpu_results.md). NVRTC controls for transformer failures are in [transformer_reference.json](transformer_reference.json).
