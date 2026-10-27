# Emmy operator and IR wishlist

These findings use emmy-ml 0.3.11 and the organisers' generator at commit
`ec8cde1943f88efea3827a95bda6085553d2d407`. Exact errors are linked from
[DL_COVERAGE.md](DL_COVERAGE.md). A tracing failure is not proof that Kernel IR
needs a new statement: some gaps belong to decomposition, scheduling or packaging.

| Observed gap | Why models need it | Suggested implementation |
|---|---|---|
| Native log-softmax: `unknown elementwise op name: 'log_softmax'` | Language-model probabilities and losses | Decompose to row max, sum of exponentials and subtraction of log-sum-exp, using existing statements. Tested log(softmax) passes normal inputs but is less stable for extreme logits. |
| Native cross-entropy: `cannot map fallback aten.cross_entropy_loss` | Token prediction and supervised classifiers | Stable log-softmax, indexed target load, then mean/sum. Dense soft-target loss passes; explicit target-gather loss currently fails on unbound `out_coord_0`. |
| Native avg/max pool2d fallback cannot map its reduced output shape | CNN downsampling | Window iteration with coordinate guards and sum/count or max accumulators. Supplied-window reductions pass but do not test extraction or padding. |
| Argmax fallback cannot map its output shape | Classification, decoding, index-returning pooling | Carry `(value, index)` through combine programs, with a defined tie rule and integer output. This differs from amax. |
| Multi-axis global average pooling: unbound `out_coord_2` | CNN heads and vision embeddings | Repair axis-coordinate mapping. Flattening spatial axes before mean already runs, so no new reduction statement is necessary. |
| Constant causal masks become `data: [null]`; one dynamic variant lacks a per-kernel dump | Causal attention | Preserve tensor-constant payloads and kernel names, or generate masks from row/column indices. Odd-size dynamic causal attention passes. |
| Fused transformer materialization fails; staged forms hit the 20-second runner timeout | Complete transformer blocks | Validate names and synchronization at kernel boundaries; preserve observable outputs and dependency order. Emmy CUDA + NVRTC also times out on the checked control case. The exact cause remains unresolved. |

Gather and scan already exist in this Emmy release. Embedding and cumsum pass the
new tests; they should not be listed as missing IR operators. The submission needed
one small backend change for an explicit integer cast on a gather index.

Further proposals, not validated by these probes:

- Native conv2d and window extraction should expose strides, dilation, groups and
  padding, then lower to guarded loads and tiled multiply-accumulate loops. Current
  convolution cases begin with prepared columns/windows.
- Top-k/sort would help decoding and mixture-of-experts routing. Small-k selection
  or block sorting networks need defined ties, index types and scratch requirements.
- Counter-based RNG with explicit seed and counter would enable reproducible dropout,
  lowered with integer arithmetic and bit operations.
- Tensor-core MMA and bf16/fp8 constructs already appear in the installed source.
  They need tested capability/precision contracts and backend coverage, rather than
  a claim that their IR types do not exist. This compiler rejects unsupported forms.
