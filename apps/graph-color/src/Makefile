CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 

#CC_FLAGS += -Xptxas=-v

#CC_FLAGS += -D'DEBUG=1'

INC = -I./ -I../../../common

all: global.o gpu-graph-color

global.o: ../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

gpu-graph-color: Makefile global.o graph_color.cpp graph_color.h graph_color_wrapper.cu graph_color_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o graph_color_wrapper.cu graph_color.cpp -o $@

clean:
	rm *.o gpu-graph-color
