extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 96) {
        int _a2_ks = _gid / 48;
        int a0_b = (_gid / 6) % 8;
        int a1_b = (_gid) % 6;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 8; a2++) {
            float in0__ar0 = x0[(a0_b * 2 + a0_u * 2) * 16 + (_a2_ks * 8 + a2)];
            float in1__bc0 = x1[(_a2_ks * 8 + a2) * 16 + (a1_b * 3 + a1_u * 3) % 16];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = x1[(_a2_ks * 8 + a2) * 16 + (a1_b * 3 + a1_u * 3 + 1) % 16];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in1__bc2 = x1[(_a2_ks * 8 + a2) * 16 + (a1_b * 3 + a1_u * 3 + 2) % 16];
            float v0__c0_2 = in0__ar0 * in1__bc2;
            acc0__c0_2 += v0__c0_2;
            float in0__ar1 = x0[(a0_b * 2 + a0_u * 2 + 1) * 16 + (_a2_ks * 8 + a2)];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__ar1 * in1__bc1;
            acc0__c1_1 += v0__c1_1;
            float v0__c1_2 = in0__ar1 * in1__bc2;
            acc0__c1_2 += v0__c1_2;
        }
        if (a1_b * 3 + a1_u * 3 + 0 < 16) {
            atomicAdd(&matmul[(a0_b * 2 + a0_u * 2) * 16 + (a1_b * 3 + a1_u * 3) % 16], acc0__c0_0);
        }
        if (a1_b * 3 + a1_u * 3 + 1 < 16) {
            atomicAdd(&matmul[(a0_b * 2 + a0_u * 2) * 16 + (a1_b * 3 + a1_u * 3 + 1) % 16], acc0__c0_1);
        }
        if (a1_b * 3 + a1_u * 3 + 2 < 16) {
            atomicAdd(&matmul[(a0_b * 2 + a0_u * 2) * 16 + (a1_b * 3 + a1_u * 3 + 2) % 16], acc0__c0_2);
        }
        if (a1_b * 3 + a1_u * 3 + 0 < 16) {
            atomicAdd(&matmul[(a0_b * 2 + a0_u * 2 + 1) * 16 + (a1_b * 3 + a1_u * 3) % 16], acc0__c1_0);
        }
        if (a1_b * 3 + a1_u * 3 + 1 < 16) {
            atomicAdd(&matmul[(a0_b * 2 + a0_u * 2 + 1) * 16 + (a1_b * 3 + a1_u * 3 + 1) % 16], acc0__c1_1);
        }
        if (a1_b * 3 + a1_u * 3 + 2 < 16) {
            atomicAdd(&matmul[(a0_b * 2 + a0_u * 2 + 1) * 16 + (a1_b * 3 + a1_u * 3 + 2) % 16], acc0__c1_2);
        }
    }
}
