NVCC ?= nvcc
# RTX 5080 = sm_120a (Consumer-Blackwell; sm_100a ist Datacenter-B100/B200)
ARCH = -gencode arch=compute_120a,code=sm_120a
OPTS = -O3 -std=c++17 -Xcompiler -Wall --use_fast_math

all: bench train architecture_test sample chat gradcheck dialogprep kgtrain kgprep profile roofline

bench: src/bench.cu src/cell.cu src/common.h Makefile
	$(NVCC) $(ARCH) $(OPTS) -o bench src/bench.cu src/cell.cu

# Shared CUDA sources are compiled in one translation unit per executable.
train: src/model.cu src/checkpoint.h src/config.h src/train_main.cu src/train.cu src/common.h src/util.h src/linalg.cu src/moe.cu src/norm.cu src/cell.cu src/emb.cu src/loss.cu src/adam.cu src/mla.cu Makefile
	$(NVCC) $(ARCH) $(OPTS) -o train src/train_main.cu -lcublas

architecture_test: src/architecture_test.cu $(wildcard src/*.h src/*.cu) Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/architecture_test.cu -lcublas

sample: src/sample.cu src/generate.h $(wildcard src/*.h src/*.cu) Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/sample.cu -lcublas

gradcheck: src/gradcheck.cu $(wildcard src/*.h src/*.cu) Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/gradcheck.cu -lcublas

kgtrain: src/kg_main.cu $(wildcard src/*.h src/*.cu) Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/kg_main.cu -lcublas

kgprep: tools/kgprep.cpp tools/json_scan.h Makefile
	$(CXX) -O3 -std=c++17 -Wall -pthread -o $@ tools/kgprep.cpp

chat: src/chat.cu src/generate.h $(wildcard src/*.h src/*.cu) Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/chat.cu -lcublas

dialogprep: tools/dialogprep.cpp tools/json_scan.h Makefile
	$(CXX) -O3 -std=c++17 -Wall -o $@ tools/dialogprep.cpp

profile: src/profile.cu src/util.h src/linalg.cu src/moe.cu Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/profile.cu -lcublas

roofline: src/roofline.cu src/common.h Makefile
	$(NVCC) $(ARCH) $(OPTS) -o $@ src/roofline.cu -lcublas

check: architecture_test train sample chat gradcheck dialogprep kgtrain kgprep
	./architecture_test
	sh tests/cli.sh
	sh tests/kg.sh

# The Zig host programs must behave like the C++ ones (needs zig).
check-zig: all
	zig build
	zig build test
	sh tests/zig.sh

clean:
	rm -f bench train selftest architecture_test sample chat gradcheck dialogprep kgtrain kgprep profile roofline

.PHONY: all check check-zig clean
