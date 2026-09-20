NVCC ?= nvcc
# RTX 5080 = sm_120a (Consumer-Blackwell; sm_100a ist Datacenter-B100/B200)
ARCH = -gencode arch=compute_120a,code=sm_120a
OPTS = -O3 -std=c++17 -Xcompiler -Wall --use_fast_math

all: bench

bench: src/bench.cu src/cell.cu src/common.h Makefile
	$(NVCC) $(ARCH) $(OPTS) -o bench src/bench.cu src/cell.cu

clean:
	rm -f bench
