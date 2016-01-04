Usage
=====
Usage: run_bfs_<STREAM>streams [dataset] < [path to graph data file]  
dataset:  
         0 - DIMACS9  
         1 - DIMACS10  
         2 - SLNDC

Datasets
========
See real and synthetic graphs in ../datasets folder

Source Files
===========
* bfs.h - definition of graph data structure and main bfs function calls
* cuda_util.h - CUDA utility functions (device initialization and error handling)
* graph_util.h - functions to parse graph datasets
* stats.h - definition of statistics to be collected for easy import in excel spreadsheet, and of function to print statistics => see stats.txt
* util.h - definition of timestamp utility function
* bfs.cpp - BFS traversal on CPU.  
  The following implementations are provided:
 1. iterative, level-based traversal
 2. recursive traversal  
     Note: for large graphs, it is necessary to change the OS setting for the stack size:
	* bash command: ulimit -s unlimited
	* csh command: set stacksize unlimited

* bfs_gpu.cu - BFS traversal on GPU  
  The following implementations are provided:
  1. flat parallelism (level-based traversal using thread-based mapping)
  2. recursive naive (with configurable number of streams)
  3. recursive hierarchical (with configurable number of streams)
* main.cu - entry point

Precompiler Variables
--------------------  
- PROFILE_GPU  
- THREADS_PER_BLOCK_FLAT  
- NUM_BLOCKS_FLAT  
- THREADS_PER_BLOCK_FLAT  
- NUM_BLOCKS  
- STREAMS  
- DEVICE  

Currently the Makefile modifies the STREAMS variable, and generates multiple binaries: run_bfs_<STREAM>streams


Know Problems
==============
- Works well for type 2 datasets (real and synthetics)
- Some problems with type 1 and 3 datasets
