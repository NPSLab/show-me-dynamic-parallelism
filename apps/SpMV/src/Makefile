CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 

#CC_FLAGS += -Xptxas=-v

#CC_FLAGS += -G -g

HALLOC_FLAGS = -I../../../lib/halloc -L../../../lib

CC_FLAGS += ${HALLOC_FLAGS}

INC = -I./ -I../../../common

all: global.o gpu-SpMV

global.o: ../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

gpu-SpMV: Makefile global.o SpMV.cpp SpMV.h SpMV_wrapper.cu SpMV_kernel.cu
	$(CC) -D'CONSOLIDATE_LEVEL=0' -D'BUFFER_ALLOCATOR=2' $(CC_FLAGS) $(INC) -lhalloc global.o SpMV_wrapper.cu SpMV.cpp -o $@

clean:
	rm *.o gpu-SpMV
