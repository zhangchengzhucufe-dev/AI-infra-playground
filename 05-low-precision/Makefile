# NVFP4 pipeline: the e2m1 conversion goes through the software encoder
# (e2m1_encode.h), so everything here runs on any sm_80+ GPU; ARCH defaults
# to native.
#   make bin/nvfp4/quantize       build one program
#   make run/nvfp4/fused_rms_quantize   the fusion study (takes a minute)

NVCC  ?= nvcc
ARCH  ?= native
FLAGS ?= -O2 -std=c++20 -I. --expt-relaxed-constexpr

bin/%: %.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -MMD -MF bin/$*.d -o $@ $<

run/%: bin/%
	./$<

clean:
	rm -rf bin

-include $(wildcard bin/*.d)

.PHONY: run/% clean
