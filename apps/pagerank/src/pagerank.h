#ifndef __PG_H__
#define __PG_H__

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "global.h"

typedef float FLOAT_T;

#define TOTAL_RANK 1000.0

typedef struct conf {
	bool verbose;
	bool debug;
	int data_set_format;
	int solution;
	int device_num;
	char *graph_file;
} CONF;

extern CONF config;

extern double init_time;
extern double d_malloc_time;
extern double h2d_memcpy_time;
extern double ker_exe_time;
extern double d2h_memcpy_time;

extern double dangling_time;

void PAGERANK_GPU();

#endif
