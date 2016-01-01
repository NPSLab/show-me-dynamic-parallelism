#include "pagerank.h"

#define N 1

using namespace std;

CONF config;

double init_time = 0;
double d_malloc_time = 0;
double h2d_memcpy_time = 0;
double ker_exe_time = 0;
double d2h_memcpy_time = 0;

double dangling_time = 0;

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage:  gpu-pagerank [option]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h      print this message\n");
	fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
	fprintf(stderr, "    --debug,-d     enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <graph_file>           import graph file\n");
	fprintf(stderr, "    --thread,-t <number of threads>    specify number of threads\n");
	fprintf(stderr, "    --format,-f <number>               specify the input format\n");
 	fprintf(stderr, "                 0 - DIMACS9\n");
	fprintf(stderr, "                 1 - DIMACS10\n");
	fprintf(stderr, "                 2 - SLNDC\n");
	fprintf(stderr, "    --solution,-s <number>             specify the solution\n");
	fprintf(stderr, "                 0 - no dynamic parallelism (thread pull, nopruning)\n");
	fprintf(stderr, "                 1 - naive dynamic parallelism baseline (per thread launch)\n");
	fprintf(stderr, "                 2 - warp-level consolidation\n");
	fprintf(stderr, "                 3 - warp-level consolidation\n");
	fprintf(stderr, "                 4 - warp-level consolidation\n");
	fprintf(stderr, "    --device,-e <number>               select the device\n");
}

void print_conf() { 
	fprintf(stderr, "\nCONFIGURATION:\n");
	if (config.graph_file) {
		fprintf(stderr, "- Graph file: %s\n", config.graph_file);
		fprintf(stderr, "- Graph format: %d\n", config.data_set_format);
 	}
	fprintf(stderr, "- Solution: %d\n", config.solution);
	if (config.verbose && config.debug) fprintf(stderr, "- verbose mode\n");
 	if (config.debug) fprintf(stderr, "- debug mode\n");
}

void init_conf() {
	config.data_set_format = 0;
	config.solution = 0;
	config.device_num = 0;
	config.verbose = false;
	config.debug = false;
}

int parse_arguments(int argc, char** argv) {
	int i = 1;
	if ( argc<2 ) {
		usage();
		return 0;
	}
	while ( i<argc ) {
		if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) {
			usage();
			return 0;
		}
		else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) {
			VERBOSE = config.verbose = 1;
		}
		else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--debug")==0 ) {
			DEBUG = config.debug = 1;
		}
		else if ( strcmp(argv[i], "-f")==0 || strcmp(argv[i], "--format")==0 ) {
			++i;
			config.data_set_format = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-s")==0 || strcmp(argv[i], "--solution")==0 ) {
			++i;
			config.solution = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-e")==0 || strcmp(argv[i], "--device")==0 ) {
			++i;
			config.device_num = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-i")==0 || strcmp(argv[i], "--import")==0 ) {
			++i;
			if (i==argc) {
				fprintf(stderr, "Graph file name missing.\n");
			}
			config.graph_file = argv[i];
		}
		++i;
	}
	
	return 1;
}

void prepare_pagerank_cpu()
{
	graph.rankArray = new FLOAT_T[noNodeTotal] ();
	for (int i=0; i<noNodeTotal; ++i ) {
		graph.rankArray[i] = (FLOAT_T)TOTAL_RANK/noNodeTotal;
	}

	graph.outdegreeArray = new int [noNodeTotal] ();
	for (int i=0; i<noNodeTotal; ++i ){
			int start = graph.vertexArray[ i ];
			int end = graph.vertexArray[ i+1 ];
			graph.outdegreeArray[i] = end - start;
	}
	return;
}

void clean_pagerank_cpu()
{
	delete [] graph.outdegreeArray;
	delete [] graph.rankArray;
}

int main(int argc, char* argv[])
{
	init_conf();
	if ( !parse_arguments(argc, argv) ) return 0;
	
	print_conf();
	
	if (config.graph_file!=NULL) {
		fp = fopen(config.graph_file, "r");
		if ( fp==NULL ) {
			fprintf(stderr, "Error: NULL file pointer.\n");
			return 1;
		}
	}
	else
		return 0;

	double start_time, end_time;
	
	start_time = gettime();
	switch(config.data_set_format) {
		case 0: readInputDIMACS9(); break;
		case 1: readInputDIMACS10(); break;
		case 2: readInputSLNDC(); break;
		default: fprintf(stderr, "Wrong code for dataset\n"); break;
	}
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "Import graph:\t\t%lf\n",end_time-start_time);
	
	start_time = gettime();
	convertCSR();
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "AdjList to CSR:\t\t%lf\n",end_time-start_time);
	
	start_time = gettime();
	invertGraph();
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "Generate inverted graph:\t\t%lf\n",end_time-start_time);
	// Prepare on CPU
	prepare_pagerank_cpu();
	/* PAGERANK on GPU */
	for (int i=0; i<N; ++i) {
		PAGERANK_GPU();
	}
	clean_pagerank_cpu();
	
	if (VERBOSE) {
		fprintf(stdout, "CUDA runtime initialization:\t\t%lf\n", init_time/N);
		fprintf(stdout, "CUDA cudaMalloc:\t\t%lf\n", d_malloc_time/N);
		fprintf(stdout, "CUDA H2D cudaMemcpy:\t\t%lf\n", h2d_memcpy_time/N);
		fprintf(stdout, "Process dangling nodes:\t\t%lf\n", dangling_time/N);
		fprintf(stdout, "CUDA kernel execution:\t\t%lf\n", ker_exe_time/N);
		fprintf(stdout, "CUDA D2H cudaMemcpy:\t\t%lf\n", d2h_memcpy_time/N);

	}	

	if ( DEBUG )
		//outputRank(stdout);
	
	clear();
	return 0;
}

