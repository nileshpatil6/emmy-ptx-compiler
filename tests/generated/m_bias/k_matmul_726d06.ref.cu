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
__launch_bounds__(256) void k_matmul_726d06(const float* b, const float* x, const float* w, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 1024;
        int a1_b = (_gid / 256) % 4;
        int a0_u = (_gid / 16) % 16;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[4096];
        __shared__ __align__(16) float _b_smem[1024];
        emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x[(a0_b * 64 + (a0_u * 16 + a1_u) / 8) * 96 + 4 * ((a0_u * 16 + a1_u) % 8)]);
        emmy_cp_async_cg(&_a_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x[(a0_b * 64 + ((a0_u * 16 + a1_u) / 8 + 32)) * 96 + 4 * ((a0_u * 16 + a1_u) % 8)]);
        for (int _fb = a0_u * 16 + a1_u; _fb < 128; _fb += 256) {
            emmy_cp_async_cg(&_b_smem[_fb * 4], &w[_fb / 4 * 64 + (a1_b * 16 + 4 * (_fb % 4))]);
        }
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 96; _ks += 32) {
            emmy_cp_async_cg(&_a_smem[(_ks / 32 + 1) % 2 * 64 * 32 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x[(a0_b * 64 + (a0_u * 16 + a1_u) / 8) * 96 + (((_ks + 32 < 96) ? (_ks + 32) : (64)) + 4 * ((a0_u * 16 + a1_u) % 8))]);
            emmy_cp_async_cg(&_a_smem[((_ks / 32 + 1) % 2 * 64 + 32) * 32 + ((a0_u * 16 + a1_u) * 4 / 32) * 32 + ((a0_u * 16 + a1_u) * 4 % 32)], &x[(a0_b * 64 + ((a0_u * 16 + a1_u) / 8 + 32)) * 96 + (((_ks + 32 < 96) ? (_ks + 32) : (64)) + 4 * ((a0_u * 16 + a1_u) % 8))]);
            for (int _fb = a0_u * 16 + a1_u; _fb < 128; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[((_ks / 32 + 1) % 2 * 32 + _fb / 4) * 16 + 4 * (_fb % 4)], &w[(((_ks + 32 < 96) ? (_ks + 32) : (64)) + _fb / 4) * 64 + (a1_b * 16 + 4 * (_fb % 4))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<1>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 32; _ki++) {
                float in0__c0_0 = _b_smem[(_ks / 32 % 2 * 32 + _ki) * 16 + (a1_b * 16 + a1_u - a1_b * 16)];
                float in1__c0_0 = _a_smem[(_ks / 32 % 2 * 64 + (a0_b * 64 + a0_u * 4 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c0_0 = in0__c0_0 * in1__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c1_0 = _a_smem[(_ks / 32 % 2 * 64 + (a0_b * 64 + a0_u * 4 + 1 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c1_0 = in0__c0_0 * in1__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float in1__c2_0 = _a_smem[(_ks / 32 % 2 * 64 + (a0_b * 64 + a0_u * 4 + 2 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c2_0 = in0__c0_0 * in1__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float in1__c3_0 = _a_smem[(_ks / 32 % 2 * 64 + (a0_b * 64 + a0_u * 4 + 3 - a0_b * 64)) * 32 + _ki];
                float acc0__v__c3_0 = in0__c0_0 * in1__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
            }
            __syncthreads();
        }
        float in2__c0_0 = b[a1_b * 16 + a1_u];
        float v1__c0_0 = acc0__c0_0 + in2__c0_0;
        add[(a0_b * 64 + a0_u * 4) * 64 + (a1_b * 16 + a1_u)] = v1__c0_0;
        float in2__c1_0 = b[a1_b * 16 + a1_u];
        float v1__c1_0 = acc0__c1_0 + in2__c1_0;
        add[(a0_b * 64 + a0_u * 4 + 1) * 64 + (a1_b * 16 + a1_u)] = v1__c1_0;
        float in2__c2_0 = b[a1_b * 16 + a1_u];
        float v1__c2_0 = acc0__c2_0 + in2__c2_0;
        add[(a0_b * 64 + a0_u * 4 + 2) * 64 + (a1_b * 16 + a1_u)] = v1__c2_0;
        float in2__c3_0 = b[a1_b * 16 + a1_u];
        float v1__c3_0 = acc0__c3_0 + in2__c3_0;
        add[(a0_b * 64 + a0_u * 4 + 3) * 64 + (a1_b * 16 + a1_u)] = v1__c3_0;
}
