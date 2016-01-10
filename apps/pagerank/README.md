Usage
=====
~~~~
Usage: gpu-pg [option]

Options:  
    --help,-h      print this message  
    --verbose,-v   basic verbosity level  
    --debug,-d     enhanced verbosity level

Other:  
    --import,-i <graph_file>           import graph file  
    --thread,-t <number of threads>    specify number of threads  
    --format,-f <number>               specify the input format  
                 0 - DIMACS9  
                 1 - DIMACS10  
                 2 - SLNDC  
    --solution,-s <number>             specify the solution  
                 0 - no dynamic parallelism (thread pull, nopruning)  
                 1 - naive dynamic parallelism baseline (per thread launch)  
                 2 - warp-level consolidation
                 3 - block-level consolidation
                 4 - grid-level consolidation
    --device,-e <number>               select the device
~~~~

Datasets
========
See [DIMACS9](http://www.dis.uniroma1.it/challenge9/), [DIMACS10] (http://www.cc.gatech.edu/dimacs10/) and [SLNDC](https://snap.stanford.edu/data/) grap    hs in `datasets` folder

Source Files
============
GPU Code
--------
* `pagerank.cpp` - entry point, include main function and other utility functions like printing help information, parsing arguments, initializing configuration and printing configuration. The main function deals with reading data and internal format conversion.
* `pagerank.h` - definition of configuration and other global variables
* `pagerank_wrapper.cu` - implementation of preparation, clean and wrapper function  
  * preparation: GPU memory allocation, data transfer, kernel configuration initialization
  * clean: GPU memory deallocation
  * wrapper: code on CPU that perform as interface between CPU and GPU. It is the wrapper of kernel launches 
* `pagerank_kernel.cu` - implementation of pagerank kernels on CPU  
  The following implementations are provided:
  1. no dynamic parallelisme  
  2. naive dynamic parallelism baseline (per thread launch)  
  3. warp-level consolidation  
  4. block-level consolidation  
  5. grid-level consolidation
  
  Note: for large graphs, it is necessary to change the OS setting for the stack size:
  * bash command: `ulimit -s unlimited`
  * csh command: `set stacksize unlimited`

Precompiler Variables  
---------------------
- `PROFILE_GPU`  

Notes
==============
- By default, the performance is measured under **10** runs and average numbers are reported. To reduce profiling time with NV Profiler, change to single run (macro `N` in `pagerank.cpp`)
