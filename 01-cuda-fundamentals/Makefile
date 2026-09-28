# Build one exercise:   make bin/env/hello     |    make run/env/hello
# ARCH defaults to native (the local card); pin it with ARCH=sm_80 on build
# nodes that have nvcc but no GPU.
#
# PTX/SASS experiments (companion to the notes in this repo):
#   make ptx/env/hello           emit PTX text and read it
#   make sassonly/env/hello      embed high-arch SASS only (won't run on lower archs)
#   make ptxonly/env/hello       embed low-arch PTX only (driver JIT picks it up)
#   ARCH_HIGH=100 make sassonly/...   if your card is already sm_90, push higher

NVCC      ?= nvcc
ARCH      ?= native
ARCH_HIGH ?= 90
# CUDA 13 dropped compute_70; 75 works on both CUDA 12 and 13
ARCH_LOW  ?= 75
FLAGS     ?= -O2 -std=c++20 -I.

bin/%: %.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $@ $<

run/%: bin/%
	./$<

ptx/%: %.cu common.h
	$(NVCC) $(FLAGS) -ptx -o $*.ptx $<
	@echo "wrote $*.ptx"

sassonly/%: %.cu common.h
	@mkdir -p $(dir bin/$*)
	$(NVCC) $(FLAGS) -gencode arch=compute_$(ARCH_HIGH),code=sm_$(ARCH_HIGH) -o bin/$*_sassonly $<
	@echo "built bin/$*_sassonly (sm_$(ARCH_HIGH) machine code only)"

ptxonly/%: %.cu common.h
	@mkdir -p $(dir bin/$*)
	$(NVCC) $(FLAGS) -gencode arch=compute_$(ARCH_LOW),code=compute_$(ARCH_LOW) -o bin/$*_ptxonly $<
	@echo "built bin/$*_ptxonly (compute_$(ARCH_LOW) PTX only)"

clean:
	rm -rf bin */*.ptx

.PHONY: run/% ptx/% sassonly/% ptxonly/% clean
