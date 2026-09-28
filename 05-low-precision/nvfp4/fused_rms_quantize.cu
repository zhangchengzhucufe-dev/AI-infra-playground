// Fused rms_norm + NVFP4 quantization.
//
// Motivation from upstream vLLM (issue #25179 / PR #36413): two PRs for
// this fusion are stuck because "the end-to-end gain is within noise and
// nobody explained where the gain went" -- this program writes the kernel
// and produces that explanation.
//
// Semantics: y = rms_norm(x) * w, quantized straight to NVFP4 (no bf16
// intermediate). rnorm = 1 / sqrt(mean(x_i^2) + eps); the per-group step
// afterwards is identical to the standalone quant.
// Byte accounting: two-step (rms writes and re-reads the intermediate)
// 6.56 B/elem, fused 2.56 B/elem, predicted speedup 2.56x. The table below
// measures per shape; the measured-vs-predicted gap is attributed per M
// range (what limits each, with ncu or arithmetic as evidence).
//
// This file provides: the two-step baseline (rms_norm_baseline_kernel
// below + the standalone quant kernel from nvfp4_quant_kernel.h), the
// fused kernel, the correctness check, and the per-shape timing harness.
// The comparison is only as good as the baseline: tune both sides to
// their best before believing the speedup (one of the upstream PRs'
// lessons).
//
// Tolerance: different sumsq reduction orders flip a tiny fraction of
// values sitting on rounding boundaries, so up to 1e-4 of bytes may differ
// (the host reference computes sumsq in double).
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "e2m1_encode.h"
#include "nvfp4_quant_kernel.h"

// Two-step baseline, first half: block-per-row rms_norm, bf16 in and out.
// Part of the fair-baseline contract: the better this is tuned, the more
// credible the comparison.
template <int BLOCK>
__global__ void rms_norm_baseline_kernel(const __nv_bfloat16* __restrict__ in,
                                         const __nv_bfloat16* __restrict__ w,
                                         __nv_bfloat16* __restrict__ out,
                                         int M, int K, float eps) {
    __shared__ float red[BLOCK / 32];
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const __nv_bfloat16* xr = in + (size_t)row * K;
        float ss = 0.f;
        for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
            float4 raw = *reinterpret_cast<const float4*>(xr + k);
            const __nv_bfloat162* h =
                reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = __bfloat1622float2(h[i]);
                ss += f.x * f.x + f.y * f.y;
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            ss = threadIdx.x < BLOCK / 32 ? red[threadIdx.x] : 0.f;
#pragma unroll
            for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
            if (threadIdx.x == 0) red[0] = ss;
        }
        __syncthreads();
        float rnorm = 1.0f / sqrtf(red[0] / K + eps);
        for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
            float4 raw = *reinterpret_cast<const float4*>(xr + k);
            float4 raww = *reinterpret_cast<const float4*>(w + k);
            const __nv_bfloat162* h =
                reinterpret_cast<const __nv_bfloat162*>(&raw);
            const __nv_bfloat162* hw =
                reinterpret_cast<const __nv_bfloat162*>(&raww);
            __nv_bfloat162 o2[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = __bfloat1622float2(h[i]);
                float2 fw = __bfloat1622float2(hw[i]);
                o2[i] = __floats2bfloat162_rn(f.x * rnorm * fw.x,
                                              f.y * rnorm * fw.y);
            }
            *reinterpret_cast<float4*>(out + (size_t)row * K + k) =
                *reinterpret_cast<float4*>(o2);
        }
        __syncthreads();
    }
}

// Fused kernel: one block per row; stage 1 reduces sumsq into rnorm,
// stage 2 quantizes group by group exactly as the standalone quant kernel
// (no bf16 intermediate written).
template <int BLOCK>
__global__ void fused_rms_nvfp4_kernel(const __nv_bfloat16* __restrict__ in,
                                       const __nv_bfloat16* __restrict__ w,
                                       uint8_t* __restrict__ dataOut,
                                       uint8_t* __restrict__ sfOut, int M,
                                       int K, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;
    const __nv_bfloat16* xr = in + (size_t)row * K;
    __shared__ float red[BLOCK / 32];

    // Stage 1: sumsq (8 elements per thread via float4 steps + shuffle tree
    // reduction)
    float ss = 0.f;
    for (int k = threadIdx.x * 8; k < K; k += BLOCK * 8) {
        float4 raw = *reinterpret_cast<const float4*>(xr + k);
        const __nv_bfloat162* h =
            reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f = __bfloat1622float2(h[i]);
            ss += f.x * f.x + f.y * f.y;
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        ss = threadIdx.x < BLOCK / 32 ? red[threadIdx.x] : 0.f;
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
        if (threadIdx.x == 0) red[0] = ss;
    }
    __syncthreads();
    float rnorm = 1.0f / sqrtf(red[0] / K + eps);

    // Stage 2: one thread per group, same order as the standalone quant
    // kernel
    int numKTiles = nvfp4_num_ktiles(K);
    int groupsPerRow = K / NVFP4_GROUP;
    for (int g = threadIdx.x; g < groupsPerRow; g += BLOCK) {
        const __nv_bfloat16* xg = xr + g * NVFP4_GROUP;
        const __nv_bfloat16* wg = w + g * NVFP4_GROUP;
        float vals[NVFP4_GROUP], amax = 0.f;
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; i++) {
            float v = __bfloat162float(xg[i]) * rnorm *
                      __bfloat162float(wg[i]);
            vals[i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
        float s = float(sf8);
        float inv = s != 0.f ? 1.0f / s : 0.0f;
        sfOut[sf_swizzled_offset(row, g, numKTiles)] = *(uint8_t*)&sf8;
        uint8_t* dst = dataOut + ((size_t)row * K / 2 + g * (NVFP4_GROUP / 2));
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; i += 2) {
            dst[i / 2] = (uint8_t)(e2m1_encode(vals[i] * inv) |
                                   (e2m1_encode(vals[i + 1] * inv) << 4));
        }
    }
}

static void launch_fused(const __nv_bfloat16* in, const __nv_bfloat16* w,
                         uint8_t* dataOut, uint8_t* sfOut, int M, int K,
                         float eps, int sms) {
    // One block per row; for few rows a single block still covers a whole
    // row (K<=8192, 256 threads x 8 elements x multiple passes); with many
    // rows the block count fills the grid naturally.
    (void)sms;
    fused_rms_nvfp4_kernel<256><<<M, 256>>>(in, w, dataOut, sfOut, M, K, eps);
}

// Fair baseline: each step tuned on its own. The rms_norm grid/block here
// is an empirical starting point; before comparing, sweep
// {256,512,1024} x {M, sms, 2sms, 4sms} on the real device and take each
// step's best (a hobbled baseline proves nothing).
static void launch_two_step(const __nv_bfloat16* in, const __nv_bfloat16* w,
                            __nv_bfloat16* mid, uint8_t* dataOut,
                            uint8_t* sfOut, int M, int K, float eps,
                            int sms) {
    int grid = M < sms * 4 ? M : sms * 4;
    rms_norm_baseline_kernel<512><<<grid, 512>>>(in, w, mid, M, K, eps);
    launch_nvfp4_quant(mid, dataOut, sfOut, M, K, sms);
}

// Host reference for the two-step baseline's first half: plain rms_norm * w,
// so the baseline's intermediate is itself validated, not just the fused
// kernel's output (a silently broken baseline would inflate the speedups).
static void host_rms_ref(const std::vector<float>& x,
                         const std::vector<float>& w, int M, int K, float eps,
                         std::vector<__nv_bfloat16>& mid) {
    mid.resize((size_t)M * K);
    for (int r = 0; r < M; r++) {
        double ss = 0;
        for (int k = 0; k < K; k++) {
            double v = x[(size_t)r * K + k];
            ss += v * v;
        }
        float rnorm = 1.0f / sqrtf((float)(ss / K) + eps);
        for (int k = 0; k < K; k++)
            mid[(size_t)r * K + k] =
                __float2bfloat16(x[(size_t)r * K + k] * rnorm * w[k]);
    }
}

static void host_ref(const std::vector<float>& x, const std::vector<float>& w,
                     int M, int K, float eps, std::vector<uint8_t>& data) {
    int numKTiles = nvfp4_num_ktiles(K);
    (void)numKTiles;
    data.assign((size_t)M * K / 2, 0);
    for (int r = 0; r < M; r++) {
        double ss = 0;
        for (int k = 0; k < K; k++) {
            double v = x[(size_t)r * K + k];
            ss += v * v;
        }
        float rnorm = 1.0f / sqrtf((float)(ss / K) + eps);
        for (int g = 0; g < K / NVFP4_GROUP; g++) {
            float vals[NVFP4_GROUP], amax = 0.f;
            for (int i = 0; i < NVFP4_GROUP; i++) {
                int k = g * NVFP4_GROUP + i;
                vals[i] = x[(size_t)r * K + k] * rnorm * w[k];
                amax = fmaxf(amax, fabsf(vals[i]));
            }
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
            float s = float(sf8);
            float inv = s != 0.f ? 1.0f / s : 0.f;
            for (int i = 0; i < NVFP4_GROUP; i += 2)
                data[(size_t)r * K / 2 + g * 8 + i / 2] =
                    (uint8_t)(e2m1_encode(vals[i + 1] * inv) << 4 |
                              e2m1_encode(vals[i] * inv));
        }
    }
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const float eps = 1e-6f;
    printf("# %-6s %-6s %10s %10s %8s\n", "M", "K", "2step_us", "fused_us",
           "speedup");
    long total_bad = 0;
    for (const auto& shape :
         {std::pair{1, 4096}, {16, 4096}, {256, 4096}, {1024, 4096},
          {4096, 4096}, {16384, 4096}, {4096, 7168}, {16384, 7168},
          {4096, 8192}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        int64_t sfB = nvfp4_sf_bytes(M, K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.f, 2.f);
        std::vector<__nv_bfloat16> hx(n), hw(K);
        std::vector<float> hxf(n), hwf(K);
        for (size_t i = 0; i < n; i++) {
            hx[i] = __float2bfloat16(dist(rng));
            hxf[i] = __bfloat162float(hx[i]);
        }
        for (int i = 0; i < K; i++) {
            hw[i] = __float2bfloat16(dist(rng) * 0.5f);
            hwf[i] = __bfloat162float(hw[i]);
        }
        __nv_bfloat16 *dx, *dw, *dmid;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dw, (size_t)K * 2));
        CUDA_CHECK(cudaMalloc(&dmid, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, sfB));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, hw.data(), (size_t)K * 2,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dsf, 0, sfB));

        launch_fused(dx, dw, dd, dsf, M, K, eps, sms);
        CUDA_CHECK_KERNEL();
        std::vector<uint8_t> gd(n / 2), rd;
        CUDA_CHECK(cudaMemcpy(gd.data(), dd, n / 2, cudaMemcpyDeviceToHost));
        host_ref(hxf, hwf, M, K, eps, rd);
        long bad = 0;
        for (size_t i = 0; i < gd.size(); i++) bad += gd[i] != rd[i];
        bool pass = bad <= (long)(gd.size() / 10000) + 1;
        total_bad += !pass;

        // validate the baseline's intermediate too (rms_norm half of
        // two-step; the quant half is quantize.cu's job). bf16 rounding of
        // the host fp32 result may differ from the kernel's fused
        // multiply, so compare in float with a small tolerance.
        launch_two_step(dx, dw, dmid, dd, dsf, M, K, eps, sms);
        CUDA_CHECK_KERNEL();
        std::vector<__nv_bfloat16> gmid(n), rmid;
        CUDA_CHECK(cudaMemcpy(gmid.data(), dmid, n * 2, cudaMemcpyDeviceToHost));
        host_rms_ref(hxf, hwf, M, K, eps, rmid);
        long mbad = 0;
        for (size_t i = 0; i < n; i++) {
            float d = __bfloat162float(gmid[i]) - __bfloat162float(rmid[i]);
            mbad += fabsf(d) > 1e-2f * (1.0f + fabsf(__bfloat162float(rmid[i])));
        }
        if (mbad) {
            printf("  baseline rms_norm output invalid (%ld / %zu elements off); "
                   "speedups below are against a broken baseline\n", mbad, n);
            total_bad++;
        }

        int iters = M >= 4096 ? 40 : 200;
        float t2 = time_avg_ms(
            [&] { launch_two_step(dx, dw, dmid, dd, dsf, M, K, eps, sms); },
            iters);
        float tf = time_avg_ms(
            [&] { launch_fused(dx, dw, dd, dsf, M, K, eps, sms); }, iters);
        printf("  %-6d %-6d %10.2f %10.2f %7.2fx %s(bad=%ld)\n", M, K,
               t2 * 1e3, tf * 1e3, t2 / tf, pass ? "PASS" : "FAIL", bad);
        cudaFree(dx); cudaFree(dw); cudaFree(dmid); cudaFree(dd);
        cudaFree(dsf);
    }
    return total_bad != 0;
}
