// End-to-end NVFP4 quantization: correctness check against a host
// reference, then bandwidth. The kernel (nvfp4_quant_kernel.h) puts one
// thread on each 16-element group: group amax -> e4m3 scale -> swizzled
// scale layout -> e2m1_encode.h packing; the host reference runs the same
// chain, so device and host agree by construction on the encoder and the
// check pins the scale math and the layouts.
//
// Correctness: every step of the quant runs the same float math in the same
// order on host and device, so results must be byte-exact, no tolerance.
// Timing prints effective bandwidth: read 2 B/elem, write 0.5 B/elem data
// + 1/16 B/elem scale factors.
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"
#include "e2m1_encode.h"
#include "nvfp4_quant_kernel.h"

static void host_ref(const std::vector<float>& x, int M, int K,
                     std::vector<uint8_t>& data, std::vector<uint8_t>& sf) {
    int numKTiles = nvfp4_num_ktiles(K);
    data.assign((size_t)M * K / 2, 0);
    sf.assign((size_t)nvfp4_sf_bytes(M, K), 0);
    for (int r = 0; r < M; r++)
        for (int g = 0; g < K / NVFP4_GROUP; g++) {
            float amax = 0.f;
            for (int i = 0; i < NVFP4_GROUP; i++)
                amax = fmaxf(amax, fabsf(x[(size_t)r * K + g * 16 + i]));
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
            float s = float(sf8);
            float inv = s != 0.f ? 1.0f / s : 0.f;
            sf[sf_swizzled_offset(r, g, numKTiles)] = *(uint8_t*)&sf8;
            for (int i = 0; i < NVFP4_GROUP; i += 2) {
                uint8_t lo = e2m1_encode(x[(size_t)r * K + g * 16 + i] * inv);
                uint8_t hi = e2m1_encode(x[(size_t)r * K + g * 16 + i + 1] * inv);
                data[(size_t)r * K / 2 + g * 8 + i / 2] = (uint8_t)(hi << 4 | lo);
            }
        }
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    long total_bad = 0;
    for (const auto& shape :
         {std::pair{128, 1024}, {200, 4096}, {4096, 7168}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        int64_t sfB = nvfp4_sf_bytes(M, K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-4.f, 4.f);
        std::vector<__nv_bfloat16> hx(n);
        std::vector<float> hxf(n);
        for (size_t i = 0; i < n; i++) {
            hx[i] = __float2bfloat16(dist(rng));
            hxf[i] = __bfloat162float(hx[i]);
        }
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, sfB));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dsf, 0, sfB));

        launch_nvfp4_quant(dx, dd, dsf, M, K, sms);
        CUDA_CHECK_KERNEL();

        std::vector<uint8_t> gd(n / 2), gsf(sfB), rd, rsf;
        CUDA_CHECK(cudaMemcpy(gd.data(), dd, n / 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gsf.data(), dsf, sfB, cudaMemcpyDeviceToHost));
        host_ref(hxf, M, K, rd, rsf);
        long bad = 0;
        for (size_t i = 0; i < gd.size(); i++) bad += gd[i] != rd[i];
        for (size_t i = 0; i < gsf.size(); i++) bad += gsf[i] != rsf[i];

        float ms = time_avg_ms(
            [&] { launch_nvfp4_quant(dx, dd, dsf, M, K, sms); },
            M >= 4096 ? 50 : 200);
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-5d K=%-5d  %s(bad=%ld)  %8.2f us  %6.0f GB/s\n", M, K,
               bad ? "FAIL" : "PASS", bad, ms * 1e3,
               effective_gbps(bytes, ms));
        total_bad += bad;
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return total_bad != 0;
}
