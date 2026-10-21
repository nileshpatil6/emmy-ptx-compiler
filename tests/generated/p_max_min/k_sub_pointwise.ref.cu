extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* y, const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 2];
        float in1__u0 = y[a0 * 2];
        float v2__u0 = fmaxf(in0__u0, in1__u0) - fminf(in0__u0, in1__u0);
        float in0__u1 = x[a0 * 2 + 1];
        float in1__u1 = y[a0 * 2 + 1];
        float v2__u1 = fmaxf(in0__u1, in1__u1) - fminf(in0__u1, in1__u1);
        float2 _vs_v2__u0_7f8948df = make_float2(v2__u0, v2__u1);
        *reinterpret_cast<float2*>(&sub[a0 * 2]) = _vs_v2__u0_7f8948df;
}
