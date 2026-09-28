// Block-wide sum reduction, three ways: interleaved pairs (divergence-heavy),
// contiguous pairs (divergence-free), and warp shuffle (no shared-memory
// reduction tree).
//
// Contract shared by the first two kernels:
//   - launch config is <<<nblocks, BLOCK>>> with BLOCK = 256;
//   - block b owns the 256 elements in[b*BLOCK] .. in[b*BLOCK + 255] and
//     writes their sum to out[b] (one thread writes it at the end of the
//     reduction, usually tid == 0);
//   - reduce in shared memory: each thread first stages its own element into
//     __shared__ float buf[BLOCK]; every add after that happens on buf;
//   - __syncthreads() before and after every round of paired adds
//
// The two kernels differ only in "which threads add each round, and where":
//   - reduce_interleaved (interleaved pairs): stride s = 1, 2, 4, ..., 128;
//     each round, threads with tid % (2*s) == 0 do buf[tid] += buf[tid + s].
//     With 8 elements:
//       s=1: buf[0]+=buf[1]  buf[2]+=buf[3]  buf[4]+=buf[5]  buf[6]+=buf[7]
//            (tids 0, 2, 4, 6 work; active threads are every other lane in the warp)
//       s=2: buf[0]+=buf[2]  buf[4]+=buf[6]   (tids 0, 4 work)
//       s=4: buf[0]+=buf[4]                   (tid 0 works)
//   - reduce_contiguous (contiguous pairs): stride s = 128, 64, ..., 1;
//     each round, threads with tid < s do buf[tid] += buf[tid + s]. Same 8 elements:
//       s=4: buf[0]+=buf[4]  buf[1]+=buf[5]  buf[2]+=buf[6]  buf[3]+=buf[7]
//            (tids 0-3 work; active threads packed at the low end)
//       s=2: buf[0]+=buf[2]  buf[1]+=buf[3]   (tids 0, 1 work)
//       s=1: buf[0]+=buf[1]                   (tid 0 works)
//   Identical number of adds in both; only how the active threads sit inside
//   a warp differs
//
// Careful: write the loop bounds in both kernels with blockDim.x (a runtime
// value), not the BLOCK macro. A compile-time constant lets the compiler fully
// unroll the loop and turn % into bit ops, which skews the timing comparison.
//
// main() includes the correctness judge (per-block sums checked against CPU
// partial sums) and times all three versions; expected output PASS per kernel
// plus the interleaved/contiguous ratio.
#include "common.h"

#define BLOCK 256

__global__ void reduce_interleaved(const float *in, float *out) {
    // Interleaved pairs: s = 1, 2, 4, ..., 128; threads with tid % (2*s) == 0 add.
    // Active threads sit every other lane; at most half a warp on one path.
    __shared__ float buf[BLOCK];
    int t = threadIdx.x;
    buf[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (t % (2 * s) == 0) {
            buf[t] += buf[t + s];
        }
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = buf[0];
}

__global__ void reduce_contiguous(const float *in, float *out) {
    // Contiguous pairs: s = 128, 64, ..., 1; threads with tid < s add.
    // Active threads packed at the low end, naturally aligned to warp
    // boundaries, so divergence stays low.
    __shared__ float buf[BLOCK];
    int t = threadIdx.x;
    buf[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            buf[t] += buf[t + s];
        }
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = buf[0];
}

// Third version: warp shuffle reduction. __shfl_down_sync exchanges
// values between registers inside a warp (syncs itself, uses no shared memory);
// each warp first reduces to a single value, the 8 warp partial sums land in
// shared, and warp 0 reduces them once more the same way.
__global__ void reduce_shuffle(const float *in, float *out) {
    __shared__ float warp_sums[BLOCK / 32];
    int t = threadIdx.x;
    float v = in[blockIdx.x * blockDim.x + t];

    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if ((t & 31) == 0) warp_sums[t >> 5] = v;
    __syncthreads();

    // 256 = 8 warps; after the first stage only 8 values remain, all inside
    // warp 0, so this stage needs intra-warp traffic only, no __syncthreads.
    if (t < 32) {
        float w = (t < BLOCK / 32) ? warp_sums[t] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, off);
        if (t == 0) out[blockIdx.x] = w;
    }
}

// ---------------- Judge and timing harness ----------------

typedef void (*reduce_fn)(const float *, float *);

static float run_one(reduce_fn fn, const char *name, const float *d_in,
                     float *d_out, float *h_out, const float *h_partial,
                     int nblocks) {
    CUDA_CHECK(cudaMemset(d_out, 0, nblocks * sizeof(float)));
    fn<<<nblocks, BLOCK>>>(d_in, d_out);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, nblocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_partial, nblocks, 1e-3f)) {
        printf("%s: FAIL\n", name);
        emit_result("block-reduce", "fail", "{}");
        exit(1);
    }

    const int reps = 200;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++) fn<<<nblocks, BLOCK>>>(d_in, d_out);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("%s: PASS  avg %.4f ms\n", name, ms);
    return ms;
}

int main() {
    const int nblocks = 4096;
    const int n = nblocks * BLOCK;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(nblocks * sizeof(float));
    float *h_partial = (float *)malloc(nblocks * sizeof(float));
    fill_random(h_in, n, 11);
    for (int b = 0; b < nblocks; b++) {
        double s = 0;
        for (int t = 0; t < BLOCK; t++) s += h_in[b * BLOCK + t];
        h_partial[b] = (float)s;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, nblocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    float ms_i = run_one(reduce_interleaved, "interleaved", d_in, d_out, h_out,
                         h_partial, nblocks);
    float ms_c = run_one(reduce_contiguous, "contiguous ", d_in, d_out, h_out,
                         h_partial, nblocks);
    // Threshold 1.5x: measured 2.22x on A100, 2.33x on V100; ~1x if both
    // versions end up as the same implementation.
    float ratio = report_speedup("interleaved / contiguous", ms_i, ms_c, 1.5f,
                                 "both versions time the same; check whether you wrote one implementation twice");

    // Shuffle version: third measurement, independent of the checks above.
    run_one(reduce_shuffle, "shuffle(warp) ", d_in, d_out, h_out, h_partial,
            nblocks);

    char metrics[192];
    snprintf(metrics, sizeof(metrics),
             "{\"interleaved_ms\":%.4f,\"contiguous_ms\":%.4f,\"ratio\":%.3f}",
             ms_i, ms_c, ratio);
    emit_result("block-reduce", "pass", metrics);
    return 0;
}
