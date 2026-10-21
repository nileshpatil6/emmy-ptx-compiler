extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float v2__u0 = cosf(in0__u0) * sinf(in0__u0);
        float v2__u1 = cosf(in0__u1) * sinf(in0__u1);
        float2 _vs_v2__u0_0d6d9abc = make_float2(v2__u0, v2__u1);
        *reinterpret_cast<float2*>(&mul[a0 * 2]) = _vs_v2__u0_0d6d9abc;
}
