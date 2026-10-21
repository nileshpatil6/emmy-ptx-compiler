extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* x1, const float* x0, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1200) {
        int a0 = _gid / 25;
        int a1 = (_gid) % 25;
        float in1__u0 = x0[a0 * 100 + a1 * 4];
        float in0__u0 = x1[a0];
        float v0__u0 = in0__u0 * in1__u0;
        float in1__u1 = x0[a0 * 100 + (a1 * 4 + 1)];
        float in0__u1 = x1[a0];
        float v0__u1 = in0__u1 * in1__u1;
        float in1__u2 = x0[a0 * 100 + (a1 * 4 + 2)];
        float in0__u2 = x1[a0];
        float v0__u2 = in0__u2 * in1__u2;
        float in1__u3 = x0[a0 * 100 + (a1 * 4 + 3)];
        float in0__u3 = x1[a0];
        float v0__u3 = in0__u3 * in1__u3;
        float4 _vs_v0__u0_7d1b531c = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 100 + a1 * 4]) = _vs_v0__u0_7d1b531c;
    }
}
