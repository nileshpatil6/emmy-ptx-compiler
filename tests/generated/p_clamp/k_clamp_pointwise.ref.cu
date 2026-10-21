extern "C" __global__
__launch_bounds__(256) void k_clamp_pointwise(const float* clamp_bound0, const float* clamp_bound1, const float* x, float* clamp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = clamp_bound0[0];
        float in2__u0 = x[a0 * 2];
        float in1__u0 = clamp_bound1[0];
        float v1__u0 = fminf(in1__u0, fmaxf(in0__u0, in2__u0));
        float in0__u1 = clamp_bound0[0];
        float in2__u1 = x[a0 * 2 + 1];
        float in1__u1 = clamp_bound1[0];
        float v1__u1 = fminf(in1__u1, fmaxf(in0__u1, in2__u1));
        float2 _vs_v1__u0_650c8aa8 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&clamp[a0 * 2]) = _vs_v1__u0_650c8aa8;
}
