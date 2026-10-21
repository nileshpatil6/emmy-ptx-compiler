extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_ccb027(const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 16;
        int a5 = (_gid) % 16;
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        float acc3__r1 = 0.0f;
        float acc5__sum__r1 = 0.0f;
        float acc1__r1 = -1e+30f;
        for (int a1 = 0; a1 < 96; a1 += 2) {
            float acc0 = 0.0f;
            for (int a2 = 0; a2 < 16; a2++) {
                float in1 = q[a0 * 16 + a2];
                float in0 = kt[a2 * 96 + a1];
                float v0 = in0 * in1;
                acc0 += v0;
            }
            float one = 1.0f;
            float acc0__gn = fmaxf(acc0, acc1);
            float acc0__alpha = expf((acc1 - acc0__gn));
            float acc0__beta = expf((acc0 - acc0__gn));
            float acc0__acc3_sa = acc0__alpha * acc3;
            float acc0__acc3_sb = acc0__beta * one;
            acc3 = acc0__acc3_sa + acc0__acc3_sb;
            float acc0__acc5__sum_sa = acc0__alpha * acc5__sum;
            float in6 = v[a1 * 16 + a5];
            float acc0__acc5__sum_sb = acc0__beta * in6;
            acc5__sum = acc0__acc5__sum_sa + acc0__acc5__sum_sb;
            acc1 = fmaxf(acc1, acc0);
            float acc0__r1 = 0.0f;
            for (int a2 = 0; a2 < 16; a2++) {
                float in1__r1 = q[a0 * 16 + a2];
                float in0__r1 = kt[a2 * 96 + (a1 + 1)];
                float v0__r1 = in0__r1 * in1__r1;
                acc0__r1 += v0__r1;
            }
            float one__r1 = 1.0f;
            float acc0__gn__r1 = fmaxf(acc0__r1, acc1__r1);
            float acc0__alpha__r1 = expf((acc1__r1 - acc0__gn__r1));
            float acc0__beta__r1 = expf((acc0__r1 - acc0__gn__r1));
            float acc0__acc3_sa__r1 = acc0__alpha__r1 * acc3__r1;
            float acc0__acc3_sb__r1 = acc0__beta__r1 * one__r1;
            acc3__r1 = acc0__acc3_sa__r1 + acc0__acc3_sb__r1;
            float acc0__acc5__sum_sa__r1 = acc0__alpha__r1 * acc5__sum__r1;
            float in6__r1 = v[(a1 + 1) * 16 + a5];
            float acc0__acc5__sum_sb__r1 = acc0__beta__r1 * in6__r1;
            acc5__sum__r1 = acc0__acc5__sum_sa__r1 + acc0__acc5__sum_sb__r1;
            acc1__r1 = fmaxf(acc1__r1, acc0__r1);
        }
        float acc1__r1__gn = fmaxf(acc1, acc1__r1);
        float acc1__r1__alpha = expf((acc1 - acc1__r1__gn));
        float acc1__r1__beta = expf((acc1__r1 - acc1__r1__gn));
        float acc1__r1__acc3_sa = acc1__r1__alpha * acc3;
        float acc1__r1__acc3_sb = acc1__r1__beta * acc3__r1;
        acc3 = acc1__r1__acc3_sa + acc1__r1__acc3_sb;
        float acc1__r1__acc5__sum_sa = acc1__r1__alpha * acc5__sum;
        float acc1__r1__acc5__sum_sb = acc1__r1__beta * acc5__sum__r1;
        acc5__sum = acc1__r1__acc5__sum_sa + acc1__r1__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r1);
        float acc5 = acc5__sum * (1.0f / acc3);
        matmul_1[a0 * 16 + a5] = acc5;
}
