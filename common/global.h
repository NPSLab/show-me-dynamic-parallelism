#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <vector>
#include <list>
#include <queue>
#include "omp.h"

#define BUFF_SIZE 1024000

#define MAX_LEVEL 9999

#define INF 1073741824	// 1024*1024*1024

#define EXIT(msg) \
	fprintf(stderr, "info: %s:%d: ", __FILE__, __LINE__); \
	fprintf(stderr, "%s", msg);	\
	exit(0);

using namespace std;

struct _GRAPH_{
	int *vertexArray;
	int *costArray;
	int *levelArray;
	int *edgeArray;
	int *weightArray;
	char *frontier;
	char *visited;
	char *update;
	int *childVertexArray;
	int *rEdgeArray;
	int *outdegreeArray;
	float *rankArray;
	int *colorArray;
};

double gettime();
double gettime_ms();


extern list<int> *adjacencyNodeList;
extern list<int> *adjacencyWeightList;

extern struct _GRAPH_ graph;
extern char buff[BUFF_SIZE];
extern int noNodeTotal;
extern int noEdgeTotal;
extern int source;
extern FILE* fp;
extern int VERBOSE;
extern int DEBUG;

extern float *bc;

int readInputDIMACS9();
int readInputDIMACS10();
int readInputSLNDC();
int convertCSR();
int invertGraph();

// Multiple DFS
void SSSP_queue_init(queue<int> &myqueue, int depth, int degree);

int outputLevel();
int outputCost(FILE *file);
int outputBC(FILE *file);
int outputRank(FILE *file);
int outputColor(FILE *file);
int clear();
#endif
