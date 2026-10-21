extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x1, const float* x0, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 480) {
        int a0 = _gid / 120;
        int a1 = (_gid / 15) % 8;
        int a2 = (_gid) % 15;
        float in1__u0 = x0[a0 * 240 + a1 * 30 + a2 * 2];
        float in0__u0 = x1[a1];
        float v0__u0 = in0__u0 + in1__u0;
        float in1__u1 = x0[a0 * 240 + a1 * 30 + (a2 * 2 + 1)];
        float in0__u1 = x1[a1];
        float v0__u1 = in0__u1 + in1__u1;
        float2 _vs_v0__u0_4416292c = make_float2(v0__u0, v0__u1);
        *reinterpret_cast<float2*>(&add[a0 * 240 + a1 * 30 + a2 * 2]) = _vs_v0__u0_4416292c;
    }
}
