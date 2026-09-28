// The first_mma.cu fp16 tile (16x16 x 16x8), loaded two ways: per-element
// smem reads + half2 packing vs ldmatrix. Both paths coexist behind the
// template switch and both must PASS against the CPU reference on several
// seeds (small integers, exact in fp16 and in the f32 accumulate, so the
// comparison is strict equality).
//   load_manual: first_mma.cu's index formulas, element by element.
//   load_ldsm:   ldmatrix.x4 for A, .x2 for B, no .trans. A is row-major
//                [16][16]; a 16B ldmatrix row covers 8 halves = half an A
//                row, so the four matrices are (row half 0/1) x (k half
//                0/1): lane l supplies &sA[8*(p&1) + (l&7)][8*(p>>1)] with
//                p = l>>3. B is staged n-major in sBn ([8][16], each row
//                32B); its fragment units are k-adjacent pairs, matrix 0 =
//                k 0-7, matrix 1 = k 8-15: lane l supplies
//                &sBn[l&7][16*(l>>3) bytes].
// The data sits in smem (main copies it in from global); both paths load
// from smem.
//
// Counting instructions in each path's smem->fragment section (cuobjdump
// -sass on the built binary): the manual path pays per-element 16-bit smem
// reads plus half2 packing and the address arithmetic; ldmatrix does the
// same job with 2 instructions. The manual path cannot avoid that work:
// each fragment register gathers two halves that are contiguous in memory
// for A (so it could read b32 there -- but B's k-pairs are not contiguous
// in the k-major layout, and keeping both paths layout-identical is the
// honest comparison).
//
// Run: make run/ldmatrix/ldmatrix (each path runs several seeds)
#include <cuda_fp16.h>
#include <random>
#include "../common.h"

// smem layout: sA is [16][16] row-major (halves). B gets two layouts --
// sBk is [16][8] (k-major, what the manual path reads) and sBn is [8][16]
// (n-major, the 16 k values of each n contiguous). The manual path is
// layout-agnostic; every ldmatrix "row address" must be 16 contiguous
// bytes, and B's fragment needs k-adjacent halves paired into a b16 --
// only the n-major layout satisfies that.
__device__ void load_manual(const __half* sA, const __half* sBk,
                            const __half* sBn, unsigned (&a)[4],
                            unsigned (&b)[2]) {
    int lane = threadIdx.x;
    int group = lane >> 2, tig = lane & 3;
    __half2* ah = reinterpret_cast<__half2*>(a);
    ah[0] = __halves2half2(sA[group * 16 + tig * 2],
                           sA[group * 16 + tig * 2 + 1]);
    ah[1] = __halves2half2(sA[(group + 8) * 16 + tig * 2],
                           sA[(group + 8) * 16 + tig * 2 + 1]);
    ah[2] = __halves2half2(sA[group * 16 + tig * 2 + 8],
                           sA[group * 16 + tig * 2 + 9]);
    ah[3] = __halves2half2(sA[(group + 8) * 16 + tig * 2 + 8],
                           sA[(group + 8) * 16 + tig * 2 + 9]);

    __half2* bh = reinterpret_cast<__half2*>(b);
    bh[0] = __halves2half2(sBk[(tig * 2) * 8 + group],
                           sBk[(tig * 2 + 1) * 8 + group]);
    bh[1] = __halves2half2(sBk[(tig * 2 + 8) * 8 + group],
                           sBk[(tig * 2 + 9) * 8 + group]);
    (void)sBn;
}

// ldmatrix loads. A: the 16x16 fp16 tile is 16 rows x 8 b16 units; the
// fragment wants matrix p as (row half p&1, k half p>>1), which is one
// .x4 with no .trans since the two halves of each register are k-adjacent
// in row-major memory. B: the fragment unit is (B[k][n], B[k+1][n]) --
// k-adjacent, same column; in sBn each n-row holds all 16 k contiguously,
// so matrix 0 = k 0-7 at byte 0 and matrix 1 = k 8-15 at byte 16 of the
// same row.
__device__ void load_ldsm(const __half* sA, const __half* sBk,
                          const __half* sBn, unsigned (&a)[4],
                          unsigned (&b)[2]) {
    int lane = threadIdx.x;
    int p = lane >> 3;
    unsigned addrA = (unsigned)__cvta_generic_to_shared(
        sA + (8 * (p & 1) + (lane & 7)) * 16 + 8 * (p >> 1));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addrA));

    unsigned addrB = (unsigned)__cvta_generic_to_shared(sBn +
                                                        (lane & 7) * 16 +
                                                        ((lane >> 3) & 1) * 8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(addrB));
    (void)sA; (void)sBk;
}

template <bool USE_LDSM>
__global__ void mma_kernel(const __half* A, const __half* B, float* D) {
    __shared__ __half sA[16 * 16], sBk[16 * 8], sBn[8 * 16];
    for (int i = threadIdx.x; i < 16 * 16; i += 32) sA[i] = A[i];
    for (int i = threadIdx.x; i < 16 * 8; i += 32) {
        sBk[i] = B[i];
        sBn[(i & 7) * 16 + (i >> 3)] = B[i];  // repack to n-major
    }
    __syncwarp();
    unsigned a[4], b[2];
    if constexpr (USE_LDSM)
        load_ldsm(sA, sBk, sBn, a, b);
    else
        load_manual(sA, sBk, sBn, a, b);
    float c[4] = {0, 0, 0, 0}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    int group = threadIdx.x >> 2, tig = threadIdx.x & 3;
    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static int run_path(bool ldsm, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 7);
    __half hA[16 * 16], hB[16 * 8];
    float ref[16 * 8] = {};
    for (int i = 0; i < 16 * 16; i++)
        hA[i] = __float2half((float)(dist(rng) - 4));
    for (int i = 0; i < 16 * 8; i++)
        hB[i] = __float2half((float)(dist(rng) - 4));
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
    if (ldsm)
        mma_kernel<true><<<1, 32>>>(dA, dB, dD);
    else
        mma_kernel<false><<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

int main() {
    long total = 0;
    for (unsigned s : {1u, 7u, 42u}) {
        int bm = run_path(false, s), bl = run_path(true, s);
        printf("seed=%-6u manual %s(%d)  ldsm %s(%d)\n", s,
               bm ? "FAIL" : "PASS", bm, bl ? "FAIL" : "PASS", bl);
        total += bm + bl;
    }
    return total != 0;
}
