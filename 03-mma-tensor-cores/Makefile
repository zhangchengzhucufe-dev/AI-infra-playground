# Build one program:   make bin/fragments/fragment_map   |   make run/fragments/fragment_map
#
# Everything in this topic runs on any sm_80+ GPU; ARCH defaults to native
# (the local card). Pin it with ARCH=sm_80 on GPU-less build nodes.
#
# Like the other tensor-core topics here, this always uses explicit -gencode
# when a non-native arch is requested, never the -arch=sm_XXXa shorthand:
# under CUDA 13.0 the shorthand expands the user-code PTX target to a bare
# compute_XXX, and ptxas then rejects arch-specific instructions with a
# misleading "instruction not supported".

NVCC  ?= nvcc
ARCH  ?= native
FLAGS ?= -O2 -std=c++20 -I. --expt-relaxed-constexpr

ifneq ($(ARCH),native)
GENCODE = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
NVCC_FLAGS = $(FLAGS) $(GENCODE)
else
NVCC_FLAGS = $(FLAGS) -arch=native
endif

bin/%: %.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -MMD -MF bin/$*.d -o $@ $<

run/%: bin/%
	./$<

ptx/%: %.cu common.h
	$(NVCC) $(FLAGS) -ptx -o $*.ptx $<
	@echo "wrote $*.ptx"

clean:
	rm -rf bin *.ptx

-include $(wildcard bin/*.d)

.PHONY: run/% ptx/% clean
