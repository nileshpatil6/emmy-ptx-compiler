extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const double* x0, const double* x1, double* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 500) {
        int a0 = _gid;
        float in0__u0 = x0[a0 * 2];
        float in1__u0 = x1[a0 * 2];
        float v0__u0 = in0__u0 + in1__u0;
        float in0__u1 = x0[a0 * 2 + 1];
        float in1__u1 = x1[a0 * 2 + 1];
        float v0__u1 = in0__u1 + in1__u1;
        add[a0 * 2] = v0__u0;
        add[a0 * 2 + 1] = v0__u1;
    }
}
