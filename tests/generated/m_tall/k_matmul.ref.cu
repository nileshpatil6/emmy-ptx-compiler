static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 4096;
        int a0_b = (_gid / 256) % 16;
        int a1_b = (_gid / 256) % 1;
        int a0_u = (_gid / 32) % 8;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c7_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[256];
        __shared__ __align__(16) float _b_smem[128];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a0_u * 32 + a1_u; _fa < 64; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[(a0_b * 64 + _fa) * 64 + (_a2_ks * 4 + _ks)]);
            }
            for (int _fb = a0_u * 32 + a1_u; _fb < 32; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[(_a2_ks * 4 + (_ks + _fb / 8)) * 32 + (a1_b * 32 + 4 * (_fb % 8))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u - a1_b * 32)];
                float in0__c0_0 = _a_smem[(a0_b * 64 + a0_u * 8 - a0_b * 64) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in0__c1_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 1 - a0_b * 64) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float in0__c2_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 2 - a0_b * 64) * 4 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float in0__c3_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 3 - a0_b * 64) * 4 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float in0__c4_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 4 - a0_b * 64) * 4 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float in0__c5_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 5 - a0_b * 64) * 4 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float in0__c6_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 6 - a0_b * 64) * 4 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float in0__c7_0 = _a_smem[(a0_b * 64 + a0_u * 8 + 7 - a0_b * 64) * 4 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
            }
            __syncthreads();
        }
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8) * 32 + (a1_b * 32 + a1_u)], acc0__c0_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 1) * 32 + (a1_b * 32 + a1_u)], acc0__c1_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 2) * 32 + (a1_b * 32 + a1_u)], acc0__c2_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 3) * 32 + (a1_b * 32 + a1_u)], acc0__c3_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 4) * 32 + (a1_b * 32 + a1_u)], acc0__c4_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 5) * 32 + (a1_b * 32 + a1_u)], acc0__c5_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 6) * 32 + (a1_b * 32 + a1_u)], acc0__c6_0);
        atomicAdd(&matmul[(a0_b * 64 + a0_u * 8 + 7) * 32 + (a1_b * 32 + a1_u)], acc0__c7_0);
}
