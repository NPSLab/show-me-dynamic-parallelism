CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2

#CC_FLAGS += --ptxas-options=-v

#CC_FLAGS += -G -g

#CC_FLAGS += -DDEBUG

#CC_FLAGS += -DGPU_PROFILE 

#CC_FLAGS += -DCPU_PROFILE

#CC_FLAGS += -DTHREASHOLD=400

HALLOC_FLAGS = -L../../../lib

CC_FLAGS += ${HALLOC_FLAGS}

INC = -I./ -I../../../common

all: global.o gpu-sssp

global.o: ../../../common/global.cpp
	${CC} ${FLAGS} ${INC} -c $^ -o $@
gpu-sssp: Makefile global.o sssp.cpp sssp_wrapper.cu sssp_kernel.cu
	$(CC) -D'CONSOLIDATE_LEVEL=0' -D'BUFFER_ALLOCATOR=2' $(CC_FLAGS) ${INC} global.o sssp.cpp sssp_wrapper.cu -o $@

clean:
	rm *.o gpu-sssp*
