// Polynomial evaluation with the 8 coefficients in constant memory vs global
// memory. poly_eval_global keeps the coefficients behind a global-memory
// pointer -- 8 VRAM reads per thread -- and stands as the baseline;
// poly_eval_const reads the same values from a __constant__ array backed by
// the dedicated constant cache. Both kernels share one signature so main()
// can run both through a single function pointer type. Horner's scheme,
// highest degree first.
// main() judges and times both; expected output PASS per kernel plus the
// global/constant ratio.
#include "common.h"

__global__ void poly_eval_global(const float *x, float *y, const float *coef,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float acc = 0.f;
        // Horner's scheme, highest degree first.
        for (int k = 7; k >= 0; k--) acc = acc * xi + coef[k];
        y[i] = acc;
    }
}

__constant__ float COEF[8];

__global__ void poly_eval_const(const float *x, float *y, const float *coef,
                                int n) {
    // Parameter coef is deliberately unused here: the 8 coefficients come from
    // the constant cache. The signature must stay identical to the global
    // version so one function pointer type runs both.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float acc = 0.f;
        for (int k = 7; k >= 0; k--) acc = acc * xi + COEF[k];
        y[i] = acc;
    }
}

// ---------------- Judge and timing harness ----------------

typedef void (*poly_fn)(const float *, float *, const float *, int);

static float run_one(poly_fn fn, const char *name, const float *d_x, float *d_y,
                     const float *d_coef, float *h_y, const float *h_ref, int n,
                     int blocks, int threads) {
    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)n * sizeof(float)));
    fn<<<blocks, threads>>>(d_x, d_y, d_coef, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (!check_close(h_y, h_ref, n, 1e-3f)) {
        printf("%s: FAIL\n", name);
        emit_result("constant-coeff", "fail", "{}");
        exit(1);
    }

    const int reps = 100;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++) fn<<<blocks, threads>>>(d_x, d_y, d_coef, n);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("%s: PASS  avg %.4f ms\n", name, ms);
    return ms;
}

int main() {
    const int n = 1 << 24;
    size_t bytes = (size_t)n * sizeof(float);
    float h_coef[8] = {1.f, -0.5f, 0.25f, -0.125f, 0.0625f, -0.03125f, 0.015625f, -0.0078125f};

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_x, n, 5);
    for (int i = 0; i < n; i++) h_x[i] = h_x[i] * 0.1f;  // keep x near [0,1) so the degree-7 polynomial cannot overflow
    for (int i = 0; i < n; i++) {
        float acc = 0.f;
        for (int k = 7; k >= 0; k--) acc = acc * h_x[i] + h_coef[k];
        h_ref[i] = acc;
    }

    float *d_x, *d_y, *d_coef;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_coef, sizeof(h_coef)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coef, h_coef, sizeof(h_coef), cudaMemcpyHostToDevice));

    // Upload the coefficient table into the __constant__ array.
    CUDA_CHECK(cudaMemcpyToSymbol(COEF, h_coef, sizeof(h_coef)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float ms_g = run_one(poly_eval_global, "global  ", d_x, d_y, d_coef, h_y,
                         h_ref, n, blocks, threads);
    float ms_c = run_one(poly_eval_const, "constant", d_x, d_y, d_coef, h_y,
                         h_ref, n, blocks, threads);
    // No speedup expected here -- a ratio near 1.00x is the normal outcome,
    // so no degradation hint is configured (warn_below = 0).
    float ratio = report_speedup("global / constant", ms_g, ms_c, 0.f, NULL);

    char metrics[192];
    snprintf(metrics, sizeof(metrics),
             "{\"global_ms\":%.4f,\"const_ms\":%.4f,\"speedup\":%.3f}", ms_g,
             ms_c, ratio);
    emit_result("constant-coeff", "pass", metrics);
    return 0;
}
