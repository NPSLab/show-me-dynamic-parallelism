CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 

#CC_FLAGS += -Xptxas=-v

INC = -I./ -I../../../common

all: global.o gpu-pg

global.o: ../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

gpu-pg: Makefile global.o pagerank.cpp pagerank.h pagerank_wrapper.cu pagerank_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o pagerank_wrapper.cu pagerank.cpp -o $@

clean:
	rm *.o gpu-pg
