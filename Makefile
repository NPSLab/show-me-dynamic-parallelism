include common/make.config

ROOT_DIR = $(shell pwd)
SRC = $(ROOT_DIR)/src
BIN = $(ROOT_DIR)/bin

all:
	cd $(ROOT_DIR)/apps/sssp      && make && cp src/gpu-sssp $(BIN) && rm -f *.o
	cd $(ROOT_DIR)/apps/SpMV      && make && cp src/gpu-SpMV $(BIN) && rm -f *.o
	cd $(ROOT_DIR)/apps/pagerank  && make && cp src/gpu-pg $(BIN)   && rm -f *.o
#	cd utility;	make;   mv stat $(BIN);	rm -f *.o;

clean:
	cd bin && rm -f *;
	cd apps/sssp; make clean;
