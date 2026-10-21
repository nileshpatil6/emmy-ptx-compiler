#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __half* x0, const __half* x1, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 500) {
        int a0 = _gid;
        __half in0__u0 = x0[a0 * 2];
        __half in1__u0 = x1[a0 * 2];
        __half v0__u0 = in0__u0 * in1__u0;
        __half in0__u1 = x0[a0 * 2 + 1];
        __half in1__u1 = x1[a0 * 2 + 1];
        __half v0__u1 = in0__u1 * in1__u1;
        __half2 _vs_v0__u0_befd7358 = __halves2half2(v0__u0, v0__u1);
        *reinterpret_cast<__half2*>(&mul[a0 * 2]) = _vs_v0__u0_befd7358;
    }
}
