// The smallest possible tensor core program: one warp issues a single
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, i.e.
// D[16x8] = A[16x16] x B[16x8] + C. A/B hold small integers (exact in fp16,
// and the f32 accumulate stays exact), so the host check against a plain
// CPU loop is strict equality.
//
// The fragment loads are written out as the index formulas from the PTX
// docs. fragment_map.cu re-derives the same scheme for the m16n8k32 fp8
// shape; make ptx/fragments/first_mma shows the generated PTX for this
// one.
//
// Run: make run/fragments/first_mma
#include <cuda_fp16.h>
#include "../common.h"

__global__ void mma_demo(const __half* A, const __half* B, float* D) {
    int lane = threadIdx.x;
    int group = lane >> 2;      // one of the 8 row groups
    int tig = lane & 3;         // thread within the group

    // A fragment: 8 fp16 per thread in 4 b32 registers.
    // Register r's two elements: (row, col) per the indices; second k half
    // in r=2,3.
    unsigned a[4];
    __half2* ah = reinterpret_cast<__half2*>(a);
    ah[0] = __halves2half2(A[(group)*16 + tig * 2], A[(group)*16 + tig * 2 + 1]);
    ah[1] = __halves2half2(A[(group + 8) * 16 + tig * 2],
                           A[(group + 8) * 16 + tig * 2 + 1]);
    ah[2] = __halves2half2(A[(group)*16 + tig * 2 + 8],
                           A[(group)*16 + tig * 2 + 9]);
    ah[3] = __halves2half2(A[(group + 8) * 16 + tig * 2 + 8],
                           A[(group + 8) * 16 + tig * 2 + 9]);

    // B fragment (col layout; B is stored [k][n] row-major in memory):
    unsigned b[2];
    __half2* bh = reinterpret_cast<__half2*>(b);
    bh[0] = __halves2half2(B[(tig * 2) * 8 + group], B[(tig * 2 + 1) * 8 + group]);
    bh[1] = __halves2half2(B[(tig * 2 + 8) * 8 + group],
                           B[(tig * 2 + 9) * 8 + group]);

    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    // D fragment: d0,d1 at row=group, d2,d3 at row=group+8.
    D[(group)*8 + tig * 2] = d[0];
    D[(group)*8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

int main() {
    __half hA[16 * 16], hB[16 * 8];
    float ref[16 * 8] = {};
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 16; k++) hA[r * 16 + k] = __float2half((r + k) % 5 - 2);
    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 8; n++) hB[k * 8 + n] = __float2half((k * n) % 3 - 1);
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 16; k++)
                ref[r * 8 + n] += __half2float(hA[r * 16 + k]) *
                                  __half2float(hB[k * 8 + n]);

    __half *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_demo<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    printf("D[0][0]=%.0f D[0][7]=%.0f D[15][0]=%.0f D[15][7]=%.0f\n", got[0],
           got[7], got[15 * 8], got[15 * 8 + 7]);
    if (bad)
        printf("FAIL: %ld mismatches\n", bad);
    else
        printf("PASS\n");
    return bad != 0;
}
