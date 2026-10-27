"""Fixed, reproducible size sweep; no schedule tuning by backend."""

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def workloads():
    specs = []

    def add(category, family, size, code, ref, knobs=""):
        specs.append(dict(id=f"{category}_bench_{family}_{size}", family=family, size=size,
                          code=code, ref=ref, rtol=1e-3, atol=1e-4, knobs=knobs))

    for n in (2**10, 2**14, 2**18, 2**22, 2**24):
        add("p", "vector_add", n, f"torch.randn({n}) + torch.randn({n})", lambda x, y: [x+y])
        add("p", "exp", n, f"torch.exp(torch.randn({n}))", lambda x: [np.exp(x.astype(np.float64)).astype(np.float32)])
    for width in (32, 128, 512, 2048, 4096):
        add("r", "row_sum", width, f"torch.randn(128, {width}).sum(-1)",
            lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], "WORK=t128,REDUCE=coop")
        add("f", "softmax", width, f"torch.softmax(torch.randn(128, {width}), -1)",
            lambda x: [softmax(x.astype(np.float64)).astype(np.float32)], "WORK=t128,REDUCE=coop")
    for n in (32, 64, 128, 256, 512, 1024):
        add("m", "matmul", n, f"torch.randn({n}, {n}) @ torch.randn({n}, {n})",
            lambda x, y: [(x.astype(np.float64) @ y).astype(np.float32)])
    for n in (16, 32, 64, 128, 256):
        add("a", "attention", n,
            f"q=torch.randn({n}, 64); k=torch.randn(64, {n}); v=torch.randn({n}, 64); torch.softmax(q @ k / 8, -1) @ v",
            lambda q, k, v: [(softmax(q.astype(np.float64) @ k / 8) @ v).astype(np.float32)])
    return specs
