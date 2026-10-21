extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a3_ks = _gid / 2048;
        int a0 = (_gid / 512) % 4;
        int a1_b = (_gid / 512) % 1;
        int a2_b = (_gid / 256) % 2;
        int a1_u = (_gid / 32) % 8;
        int a2_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float acc0__c8_0 = 0.0f;
        float acc0__c9_0 = 0.0f;
        float acc0__c10_0 = 0.0f;
        float acc0__c11_0 = 0.0f;
        #pragma unroll
        for (int a3 = 0; a3 < 1; a3++) {
            float in0__ar0 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12) % 32 * 64 + (_a3_ks + a3)];
            float in1__bc0 = x1[a0 * 3072 + (_a3_ks + a3) * 48 + (a2_b * 32 + a2_u) % 48];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in0__ar1 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 1) % 32 * 64 + (_a3_ks + a3)];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float in0__ar2 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 2) % 32 * 64 + (_a3_ks + a3)];
            float v0__c2_0 = in0__ar2 * in1__bc0;
            acc0__c2_0 += v0__c2_0;
            float in0__ar3 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 3) % 32 * 64 + (_a3_ks + a3)];
            float v0__c3_0 = in0__ar3 * in1__bc0;
            acc0__c3_0 += v0__c3_0;
            float in0__ar4 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 4) % 32 * 64 + (_a3_ks + a3)];
            float v0__c4_0 = in0__ar4 * in1__bc0;
            acc0__c4_0 += v0__c4_0;
            float in0__ar5 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 5) % 32 * 64 + (_a3_ks + a3)];
            float v0__c5_0 = in0__ar5 * in1__bc0;
            acc0__c5_0 += v0__c5_0;
            float in0__ar6 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 6) % 32 * 64 + (_a3_ks + a3)];
            float v0__c6_0 = in0__ar6 * in1__bc0;
            acc0__c6_0 += v0__c6_0;
            float in0__ar7 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 7) % 32 * 64 + (_a3_ks + a3)];
            float v0__c7_0 = in0__ar7 * in1__bc0;
            acc0__c7_0 += v0__c7_0;
            float in0__ar8 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 8) % 32 * 64 + (_a3_ks + a3)];
            float v0__c8_0 = in0__ar8 * in1__bc0;
            acc0__c8_0 += v0__c8_0;
            float in0__ar9 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 9) % 32 * 64 + (_a3_ks + a3)];
            float v0__c9_0 = in0__ar9 * in1__bc0;
            acc0__c9_0 += v0__c9_0;
            float in0__ar10 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 10) % 32 * 64 + (_a3_ks + a3)];
            float v0__c10_0 = in0__ar10 * in1__bc0;
            acc0__c10_0 += v0__c10_0;
            float in0__ar11 = x0[a0 * 2048 + (a1_b * 96 + a1_u * 12 + 11) % 32 * 64 + (_a3_ks + a3)];
            float v0__c11_0 = in0__ar11 * in1__bc0;
            acc0__c11_0 += v0__c11_0;
        }
        if (a1_b * 96 + a1_u * 12 + 0 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c0_0);
        }
        if (a1_b * 96 + a1_u * 12 + 1 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 1) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c1_0);
        }
        if (a1_b * 96 + a1_u * 12 + 2 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 2) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c2_0);
        }
        if (a1_b * 96 + a1_u * 12 + 3 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 3) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c3_0);
        }
        if (a1_b * 96 + a1_u * 12 + 4 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 4) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c4_0);
        }
        if (a1_b * 96 + a1_u * 12 + 5 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 5) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c5_0);
        }
        if (a1_b * 96 + a1_u * 12 + 6 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 6) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c6_0);
        }
        if (a1_b * 96 + a1_u * 12 + 7 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 7) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c7_0);
        }
        if (a1_b * 96 + a1_u * 12 + 8 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 8) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c8_0);
        }
        if (a1_b * 96 + a1_u * 12 + 9 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 9) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c9_0);
        }
        if (a1_b * 96 + a1_u * 12 + 10 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 10) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c10_0);
        }
        if (a1_b * 96 + a1_u * 12 + 11 < 32 && a2_b * 32 + a2_u * 1 + 0 < 48) {
            atomicAdd(&matmul[a0 * 1536 + (a1_b * 96 + a1_u * 12 + 11) % 32 * 48 + (a2_b * 32 + a2_u) % 48], acc0__c11_0);
        }
}
