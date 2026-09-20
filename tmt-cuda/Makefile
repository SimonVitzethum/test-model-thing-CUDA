NVCC ?= nvcc
# RTX 5080 = sm_120a (Consumer-Blackwell; sm_100a ist Datacenter-B100/B200)
ARCH = -gencode arch=compute_120a,code=sm_120a
OPTS = -O3 -std=c++17 -Xcompiler -Wall --use_fast_math

all: bench train

bench: src/bench.cu src/cell.cu src/common.h Makefile
	$(NVCC) $(ARCH) $(OPTS) -o bench src/bench.cu src/cell.cu

# S2: alles in einer TU (inline-Host-Helper), kein Link-Aerger
train: src/train_main.cu src/train.cu src/util.h src/linalg.cu src/moe.cu src/norm.cu src/cell.cu src/emb.cu src/loss.cu src/adam.cu Makefile
	$(NVCC) $(ARCH) $(OPTS) -o train src/train_main.cu -lcublas

clean:
	rm -f bench train
