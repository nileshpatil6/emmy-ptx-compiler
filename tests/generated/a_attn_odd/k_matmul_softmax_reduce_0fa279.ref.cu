extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_0fa279(const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 260) {
        int a0_b = _gid / 20;
        int a5_b = (_gid) % 20;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        float in0 = mul_c1[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        float acc3__c2_0 = 0.0f;
        float acc5__sum__c2_0 = 0.0f;
        float acc1__c2_0 = -1e+30f;
        float acc3__c2_1 = 0.0f;
        float acc5__sum__c2_1 = 0.0f;
        float acc1__c2_1 = -1e+30f;
        float acc3__c3_0 = 0.0f;
        float acc5__sum__c3_0 = 0.0f;
        float acc1__c3_0 = -1e+30f;
        float acc3__c3_1 = 0.0f;
        float acc5__sum__c3_1 = 0.0f;
        float acc1__c3_1 = -1e+30f;
        for (int a1 = 0; a1 < 70; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 24; a2++) {
                float in2__ar0 = q[(a0_b * 4 + a0_u * 4) % 50 * 24 + a2];
                float in1__ar0 = kt[a2 * 70 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 24; a2++) {
                float in2__ar1 = q[(a0_b * 4 + a0_u * 4 + 1) % 50 * 24 + a2];
                float in1__ar1 = kt[a2 * 70 + a1];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 24; a2++) {
                float in2__ar2 = q[(a0_b * 4 + a0_u * 4 + 2) % 50 * 24 + a2];
                float in1__ar2 = kt[a2 * 70 + a1];
                float v0__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v0__ar2;
            }
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 24; a2++) {
                float in2__ar3 = q[(a0_b * 4 + a0_u * 4 + 3) % 50 * 24 + a2];
                float in1__ar3 = kt[a2 * 70 + a1];
                float v0__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v0__ar3;
            }
            float v1__c0_0 = acc0__ar0 * in0;
            float one__c0_0 = 1.0f;
            float v1__gn__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__alpha__c0_0 = expf((acc1__c0_0 - v1__gn__c0_0));
            float v1__beta__c0_0 = expf((v1__c0_0 - v1__gn__c0_0));
            float v1__acc3_sa__c0_0 = acc3__c0_0 * v1__alpha__c0_0;
            float v1__acc3_sb__c0_0 = one__c0_0 * v1__beta__c0_0;
            acc3__c0_0 = v1__acc3_sa__c0_0 + v1__acc3_sb__c0_0;
            float v1__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v1__alpha__c0_0;
            float2 _v_in7__bc0 = *reinterpret_cast<const float2*>(&v[a1 * 40 + (a5_b * 2 + a5_u * 2)]);
            float in7__bc0 = _v_in7__bc0.x;
            float in7__bc1 = _v_in7__bc0.y;
            float v1__acc5__sum_sb__c0_0 = in7__bc0 * v1__beta__c0_0;
            acc5__sum__c0_0 = v1__acc5__sum_sa__c0_0 + v1__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__c0_1 = acc0__ar0 * in0;
            float one__c0_1 = 1.0f;
            float v1__gn__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__alpha__c0_1 = expf((acc1__c0_1 - v1__gn__c0_1));
            float v1__beta__c0_1 = expf((v1__c0_1 - v1__gn__c0_1));
            float v1__acc3_sa__c0_1 = acc3__c0_1 * v1__alpha__c0_1;
            float v1__acc3_sb__c0_1 = one__c0_1 * v1__beta__c0_1;
            acc3__c0_1 = v1__acc3_sa__c0_1 + v1__acc3_sb__c0_1;
            float v1__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v1__alpha__c0_1;
            float v1__acc5__sum_sb__c0_1 = in7__bc1 * v1__beta__c0_1;
            acc5__sum__c0_1 = v1__acc5__sum_sa__c0_1 + v1__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__c1_0 = acc0__ar1 * in0;
            float one__c1_0 = 1.0f;
            float v1__gn__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__alpha__c1_0 = expf((acc1__c1_0 - v1__gn__c1_0));
            float v1__beta__c1_0 = expf((v1__c1_0 - v1__gn__c1_0));
            float v1__acc3_sa__c1_0 = acc3__c1_0 * v1__alpha__c1_0;
            float v1__acc3_sb__c1_0 = one__c1_0 * v1__beta__c1_0;
            acc3__c1_0 = v1__acc3_sa__c1_0 + v1__acc3_sb__c1_0;
            float v1__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v1__alpha__c1_0;
            float v1__acc5__sum_sb__c1_0 = in7__bc0 * v1__beta__c1_0;
            acc5__sum__c1_0 = v1__acc5__sum_sa__c1_0 + v1__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__c1_1 = acc0__ar1 * in0;
            float one__c1_1 = 1.0f;
            float v1__gn__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__alpha__c1_1 = expf((acc1__c1_1 - v1__gn__c1_1));
            float v1__beta__c1_1 = expf((v1__c1_1 - v1__gn__c1_1));
            float v1__acc3_sa__c1_1 = acc3__c1_1 * v1__alpha__c1_1;
            float v1__acc3_sb__c1_1 = one__c1_1 * v1__beta__c1_1;
            acc3__c1_1 = v1__acc3_sa__c1_1 + v1__acc3_sb__c1_1;
            float v1__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v1__alpha__c1_1;
            float v1__acc5__sum_sb__c1_1 = in7__bc1 * v1__beta__c1_1;
            acc5__sum__c1_1 = v1__acc5__sum_sa__c1_1 + v1__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__c2_0 = acc0__ar2 * in0;
            float one__c2_0 = 1.0f;
            float v1__gn__c2_0 = fmaxf(acc1__c2_0, v1__c2_0);
            float v1__alpha__c2_0 = expf((acc1__c2_0 - v1__gn__c2_0));
            float v1__beta__c2_0 = expf((v1__c2_0 - v1__gn__c2_0));
            float v1__acc3_sa__c2_0 = acc3__c2_0 * v1__alpha__c2_0;
            float v1__acc3_sb__c2_0 = one__c2_0 * v1__beta__c2_0;
            acc3__c2_0 = v1__acc3_sa__c2_0 + v1__acc3_sb__c2_0;
            float v1__acc5__sum_sa__c2_0 = acc5__sum__c2_0 * v1__alpha__c2_0;
            float v1__acc5__sum_sb__c2_0 = in7__bc0 * v1__beta__c2_0;
            acc5__sum__c2_0 = v1__acc5__sum_sa__c2_0 + v1__acc5__sum_sb__c2_0;
            acc1__c2_0 = fmaxf(acc1__c2_0, v1__c2_0);
            float v1__c2_1 = acc0__ar2 * in0;
            float one__c2_1 = 1.0f;
            float v1__gn__c2_1 = fmaxf(acc1__c2_1, v1__c2_1);
            float v1__alpha__c2_1 = expf((acc1__c2_1 - v1__gn__c2_1));
            float v1__beta__c2_1 = expf((v1__c2_1 - v1__gn__c2_1));
            float v1__acc3_sa__c2_1 = acc3__c2_1 * v1__alpha__c2_1;
            float v1__acc3_sb__c2_1 = one__c2_1 * v1__beta__c2_1;
            acc3__c2_1 = v1__acc3_sa__c2_1 + v1__acc3_sb__c2_1;
            float v1__acc5__sum_sa__c2_1 = acc5__sum__c2_1 * v1__alpha__c2_1;
            float v1__acc5__sum_sb__c2_1 = in7__bc1 * v1__beta__c2_1;
            acc5__sum__c2_1 = v1__acc5__sum_sa__c2_1 + v1__acc5__sum_sb__c2_1;
            acc1__c2_1 = fmaxf(acc1__c2_1, v1__c2_1);
            float v1__c3_0 = acc0__ar3 * in0;
            float one__c3_0 = 1.0f;
            float v1__gn__c3_0 = fmaxf(acc1__c3_0, v1__c3_0);
            float v1__alpha__c3_0 = expf((acc1__c3_0 - v1__gn__c3_0));
            float v1__beta__c3_0 = expf((v1__c3_0 - v1__gn__c3_0));
            float v1__acc3_sa__c3_0 = acc3__c3_0 * v1__alpha__c3_0;
            float v1__acc3_sb__c3_0 = one__c3_0 * v1__beta__c3_0;
            acc3__c3_0 = v1__acc3_sa__c3_0 + v1__acc3_sb__c3_0;
            float v1__acc5__sum_sa__c3_0 = acc5__sum__c3_0 * v1__alpha__c3_0;
            float v1__acc5__sum_sb__c3_0 = in7__bc0 * v1__beta__c3_0;
            acc5__sum__c3_0 = v1__acc5__sum_sa__c3_0 + v1__acc5__sum_sb__c3_0;
            acc1__c3_0 = fmaxf(acc1__c3_0, v1__c3_0);
            float v1__c3_1 = acc0__ar3 * in0;
            float one__c3_1 = 1.0f;
            float v1__gn__c3_1 = fmaxf(acc1__c3_1, v1__c3_1);
            float v1__alpha__c3_1 = expf((acc1__c3_1 - v1__gn__c3_1));
            float v1__beta__c3_1 = expf((v1__c3_1 - v1__gn__c3_1));
            float v1__acc3_sa__c3_1 = acc3__c3_1 * v1__alpha__c3_1;
            float v1__acc3_sb__c3_1 = one__c3_1 * v1__beta__c3_1;
            acc3__c3_1 = v1__acc3_sa__c3_1 + v1__acc3_sb__c3_1;
            float v1__acc5__sum_sa__c3_1 = acc5__sum__c3_1 * v1__alpha__c3_1;
            float v1__acc5__sum_sb__c3_1 = in7__bc1 * v1__beta__c3_1;
            acc5__sum__c3_1 = v1__acc5__sum_sa__c3_1 + v1__acc5__sum_sb__c3_1;
            acc1__c3_1 = fmaxf(acc1__c3_1, v1__c3_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a0_b * 4 + a0_u * 4 + 0 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4) % 50 * 40 + (a5_b * 2 + a5_u * 2)] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a0_b * 4 + a0_u * 4 + 0 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4) % 50 * 40 + (a5_b * 2 + a5_u * 2 + 1)] = acc5__c0_1;
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a0_b * 4 + a0_u * 4 + 1 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) % 50 * 40 + (a5_b * 2 + a5_u * 2)] = acc5__c1_0;
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a0_b * 4 + a0_u * 4 + 1 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) % 50 * 40 + (a5_b * 2 + a5_u * 2 + 1)] = acc5__c1_1;
        }
        float acc5__c2_0 = acc5__sum__c2_0 * (1.0f / acc3__c2_0);
        if (a0_b * 4 + a0_u * 4 + 2 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) % 50 * 40 + (a5_b * 2 + a5_u * 2)] = acc5__c2_0;
        }
        float acc5__c2_1 = acc5__sum__c2_1 * (1.0f / acc3__c2_1);
        if (a0_b * 4 + a0_u * 4 + 2 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) % 50 * 40 + (a5_b * 2 + a5_u * 2 + 1)] = acc5__c2_1;
        }
        float acc5__c3_0 = acc5__sum__c3_0 * (1.0f / acc3__c3_0);
        if (a0_b * 4 + a0_u * 4 + 3 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) % 50 * 40 + (a5_b * 2 + a5_u * 2)] = acc5__c3_0;
        }
        float acc5__c3_1 = acc5__sum__c3_1 * (1.0f / acc3__c3_1);
        if (a0_b * 4 + a0_u * 4 + 3 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) % 50 * 40 + (a5_b * 2 + a5_u * 2 + 1)] = acc5__c3_1;
        }
    }
}
