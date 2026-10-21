extern "C" __global__
__launch_bounds__(256) void k_sin_pointwise(const float* mul_c1, const float* x, float* sin) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = mul_c1[0];
        float in1__u0 = x[a0 * 2];
        float v1__u0 = sinf((in0__u0 * in1__u0));
        float in0__u1 = mul_c1[0];
        float in1__u1 = x[a0 * 2 + 1];
        float v1__u1 = sinf((in0__u1 * in1__u1));
        float2 _vs_v1__u0_370495d0 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&sin[a0 * 2]) = _vs_v1__u0_370495d0;
}
