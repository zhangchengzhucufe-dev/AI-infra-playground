# The thin-GEMM study: cuBLAS bf16 across 63 real LLM shapes. It links
# cuBLAS (the study subject) and the driver API, and runs on any GPU.
#   make bin/thin-gemm/thin_gemm && ./bin/thin-gemm/thin_gemm 25 300
# (args are the peak TFLOPS / GB/s references; 25/300 fits an RTX 3060
# laptop, use the right numbers for your card)

NVCC  ?= nvcc
ARCH  ?= native
FLAGS ?= -O2 -std=c++20 -I. --expt-relaxed-constexpr

bin/thin-gemm/thin_gemm: thin-gemm/thin_gemm.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -lcublas -lcuda -o $@ $<

run/thin-gemm: bin/thin-gemm/thin_gemm
	./bin/thin-gemm/thin_gemm

clean:
	rm -rf bin

.PHONY: run/thin-gemm clean
