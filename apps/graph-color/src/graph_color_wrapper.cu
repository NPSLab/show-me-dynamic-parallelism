#include <stdio.h>
#include <cuda.h>
#include "graph_color.h"

#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000

#define WARP_SIZE 32

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 0
#endif

#include "graph_color_kernel.cu"

int *d_vertexArray;
int *d_edgeArray;
int *d_colorArray;
int *d_work_queue;

unsigned int *d_queue_length;
unsigned int *d_nonstop;

dim3 dimGrid(1,1,1);	// thread+bitmap
dim3 dimBlock(1,1,1);	
int maxDegreeT = 128;	// thread/block, thread+queue
dim3 dimGridT(1,1,1);
dim3 dimBlockT(maxDegreeT,1,1);

int maxDegreeB = 32;
dim3 dimBGrid(1,1,1);	// block+bitmap
dim3 dimBBlock(maxDegreeB,1,1);		
dim3 dimGridB(1,1,1);
dim3 dimBlockB(maxDegreeB,1,1); // block+queue

//int *queue = new int [queue_max_length];
unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;

double start_time, end_time;
	
inline void cudaCheckError(int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
		exit(1);
	}
}

void prepare_gpu()
{	
	start_time = gettime();
	cudaFree(NULL);
	end_time = gettime();
	if (VERBOSE) {
		fprintf(stderr, "CUDA runtime initialization:\t\t%lf\n",end_time-start_time);
	}
	start_time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(config.device_num) );
	end_time = gettime();
	if (VERBOSE) {
		fprintf(stderr, "Choose CUDA device: %d\n", config.device_num);
		fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
	}
	/* Configuration for thread+bitmap*/	
	if ( noNodeTotal > maxDegreeT ){
		dimGrid.x = noNodeTotal / maxDegreeT + 1;
		dimBlock.x = maxDegreeT;
	}
	else {
		dimGrid.x = 1;
		dimBlock.x = noNodeTotal;
	}
	/* Configuration for block+bitmap */
	if ( noNodeTotal > MAXDIMGRID ){
		dimBGrid.x = MAXDIMGRID;
		dimBGrid.y = noNodeTotal / MAXDIMGRID + 1;
	}
	else {
		dimBGrid.x = noNodeTotal;
	}
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_colorArray, sizeof(int)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMalloc:\t\t%lf\n",end_time-start_time);

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_colorArray, graph.colorArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMemcpy:\t\t%lf\n", end_time-start_time);
}

void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_colorArray);
	cudaFree(d_nonstop);
}

void gclr_nopruning_gpu()
{	
	/* prepare GPU */

	nonstop = 1;
	int color_type = 1;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
		gclr_bitmap_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_colorArray,
												  d_nonstop, color_type, noNodeTotal );
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}   

	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
}

void gclr_np_naive_gpu()
{
	/* prepare GPU */

	/* initialize the unordered working set */
	nonstop = 1;
	int color_type = 1;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
		gclr_bitmap_multidp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_colorArray, 
															color_type, noNodeTotal);
		color_type++;
		check_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_nonstop, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		//if ( DEBUG )
		if ( DEBUG && (color_type-1)%100==0 )
			fprintf(stderr, "Iteration: %d\n", color_type-1);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);

}

void gclr_np_consolidate_gpu()
{
	int *d_buffer;
	unsigned int *d_buf_size;
	unsigned int *d_count;

	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buf_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	nonstop = 1;
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		switch (config.solution) {
		case 2:
			gclr_bitmap_cons_warp_dp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																	noNodeTotal, d_buffer);
			break;
		case 3:
			gclr_bitmap_cons_block_dp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																	noNodeTotal, d_buffer);
			break;
		case 4:
			cudaCheckError( __LINE__, cudaMemset(d_buf_size, 0, sizeof(unsigned int)));
			cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
			gclr_bitmap_cons_grid_dp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																	noNodeTotal, d_buffer, d_buf_size, d_count);
			break;
		default:
			printf("Unsupported solution\n");
			exit(0);
		}

		check_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_nonstop, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG && (color_type-1)%100==0)
			fprintf(stderr, "Iteration: %d\n", color_type-1);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_buffer);
}

void GRAPH_COLOR_GPU()
{
	prepare_gpu();
#ifdef GPU_PROFILE
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif
	start_time = gettime();
	switch (config.solution) {
		case 0:  gclr_nopruning_gpu();	//
			break;
		case 1:  gclr_np_naive_gpu();	//
			break;
		case 2:
		case 3:
		case 4:  gclr_np_consolidate_gpu();	//
			break;
		default:
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
	fprintf(stderr, "Execution time:\t\t%lf\n", end_time-start_time);
	cudaCheckError( __LINE__, cudaMemcpy( graph.colorArray, d_colorArray, sizeof(int)*noNodeTotal, cudaMemcpyDeviceToHost) );
#ifdef GPU_PROFILE
	gpu_statistics<<<1,1>>>(config.solution);
#endif
	clean_gpu();
}

