extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float v3__u0 = fabsf(in0__u0) + fmaxf(0.0f, (0.0f - in0__u0));
        float v3__u1 = fabsf(in0__u1) + fmaxf(0.0f, (0.0f - in0__u1));
        float2 _vs_v3__u0_1af82dbe = make_float2(v3__u0, v3__u1);
        *reinterpret_cast<float2*>(&add[a0 * 2]) = _vs_v3__u0_1af82dbe;
}
