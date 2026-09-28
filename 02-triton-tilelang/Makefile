# The only CUDA file here is the naive matmul baseline in matmul_ladder/;
# the Triton/TileLang rungs of the ladder are plain Python (see that dir's README).
NVCC ?= nvcc
ARCH ?= native

bin/matmul_ladder/naive: matmul_ladder/naive.cu matmul_ladder/common.h
	@mkdir -p $(dir $@)
	$(NVCC) -O2 -std=c++20 -I matmul_ladder -arch=$(ARCH) -o $@ $<

run/naive: bin/matmul_ladder/naive
	./bin/matmul_ladder/naive

clean:
	rm -rf bin

.PHONY: run/naive clean
