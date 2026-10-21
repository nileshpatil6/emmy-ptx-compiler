extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* sub_c1, const float* mul_c1, const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 259) {
        int a0 = _gid;
        float in0__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 3];
        float in1__u0 = sub_c1[0];
        float v1__u0 = (in0__u0 * in2__u0) - in1__u0;
        float in0__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 3 + 1];
        float in1__u1 = sub_c1[0];
        float v1__u1 = (in0__u1 * in2__u1) - in1__u1;
        float in0__u2 = mul_c1[0];
        float in2__u2 = x[a0 * 3 + 2];
        float in1__u2 = sub_c1[0];
        float v1__u2 = (in0__u2 * in2__u2) - in1__u2;
        sub[a0 * 3] = v1__u0;
        sub[a0 * 3 + 1] = v1__u1;
        sub[a0 * 3 + 2] = v1__u2;
    }
}
