#include <stdio.h>
#include <cuda.h>
#include "sssp.h"

#include "halloc.h"

#define INF 1073741824	// 1024*1024*1024
#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 0
#endif

#include "sssp_kernel.cu"

int *d_vertexArray;
int *d_costArray;
int *d_edgeArray;
int *d_weightArray;
int *d_work_queue;
int *d_buffer;
char *d_update;
unsigned int *d_queue_length;
unsigned int *d_bSize;

dim3 dimGrid(1,1,1);	// thread+bitmap
dim3 dimBlock(1,1,1);
int maxDegreeB = NESTED_BLOCK_SIZE;	
dim3 dimBGrid(1,1,1);	// block+bitmap
dim3 dimBBlock(maxDegreeB,1,1);
int maxDegreeT = THREADS_PER_BLOCK;	// thread/block, thread+queue
dim3 dimGridT(1,1,1);
dim3 dimBlockT(maxDegreeT,1,1);

dim3 dimGridB(1,1,1);	// block+queue
dim3 dimBlockB(maxDegreeB,1,1);

unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;

unsigned int bSize = 0;
unsigned int iteration = 0;

inline void cudaCheckError(int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
		exit(1);
	}
}

void prepare_gpu()
{
	cudaDeviceReset();
	start_time = gettime();
	cudaFree(NULL);
	end_time = gettime();
	init_time += end_time - start_time;
	
	start_time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(config.device_num) );
	end_time = gettime();
	if (DEBUG) {
		fprintf(stderr, "Choose CUDA device: %d\n", config.device_num);
		//fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
	}

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

	/* initialization */
	for (int i=0; i<noNodeTotal; i++ ) {
		graph.costArray[i] = INF;
	}
	graph.update[source] = 1;
	graph.costArray[source] = 0;
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_costArray, sizeof(int)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_weightArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_update, sizeof(char)*noNodeTotal ) );
	end_time = gettime();
	d_malloc_time += end_time - start_time;
	
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );		
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_costArray, graph.costArray, sizeof(int)*noNodeTotal,cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_weightArray, graph.weightArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_update, graph.update,  sizeof(char)*noNodeTotal, cudaMemcpyHostToDevice) );
	end_time = gettime();
	h2d_memcpy_time += end_time - start_time;
	
	/* Initialize GPU allocator */
	start_time = gettime_ms();
#if BUFFER_ALLOCATOR == 0 // default
 	size_t limit = 0;
 	if (DEBUG) {
		cudaCheckError( __LINE__, cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
 		printf("cudaLimistMallocHeapSize: %u\n", (unsigned)limit);
 	}
 	limit = GM_BUFF_SIZE*sizeof(unsigned int); // don't understand why need multiplied by 10 (otherwise crash)
 	cudaCheckError( __LINE__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
 	if (DEBUG) {
 		cudaCheckError( __LINE__, cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
 		printf("cudaLimistMallocHeapSize: %u\n", (unsigned)limit);
	}
#elif BUFFER_ALLOCATOR == 1 // halloc
	size_t memory = GM_BUFF_SIZE*16;
	ha_init(halloc_opts_t(memory));
#else

#endif
 	end_time = gettime_ms();
 	//fprintf(stderr, "Set Heap Size:\t\t%.2lf ms.\n", end_time-start_time);

}

void clean_gpu()
{
	cudaCheckError( __LINE__, cudaFree(d_vertexArray) );
	cudaCheckError( __LINE__, cudaFree(d_costArray) );
	cudaCheckError( __LINE__, cudaFree(d_edgeArray) );
	cudaCheckError( __LINE__, cudaFree(d_weightArray) );
	cudaCheckError( __LINE__, cudaFree(d_update) );
}

void sssp_tqueue_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	
	iteration = 0;
	unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, noNodeTotal, d_work_queue, 
														d_queue_length, queue_max_length);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );		
	
	while( queue_length!=0 ) {
		// unorder+thread+queue
		/* Dynamic kernel configuration */
		if (queue_length<=maxDegreeT){
			dimGridT.x = 1;
		}
		else if (queue_length<=maxDegreeT*MAXDIMGRID){
			dimGridT.x = queue_length/maxDegreeT+1;
		}
		else{
			printf("Too many elements in queue\n");
			exit(0);	
		}		
		unorder_threadQueue_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray, 
															d_weightArray, d_update, noNodeTotal,
															d_work_queue, d_queue_length);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)));
		unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, noNodeTotal, d_work_queue, 
															d_queue_length, queue_max_length);
		cudaCheckError( __LINE__, cudaMemcpy(&queue_length,d_queue_length,sizeof(unsigned int), cudaMemcpyDeviceToHost));
#ifdef CPU_PROFILE
		//printf("Working set size is %d\n", queue_length);
		//printf("%d\n", queue_length);
#endif
		iteration++;
	}

	cudaCheckError( __LINE__, cudaFree(d_work_queue) );
	cudaCheckError( __LINE__, cudaFree(d_queue_length) );
}

void sssp_np_naive_gpu()
{

	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	
	unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, noNodeTotal, d_work_queue, 
														d_queue_length, queue_max_length);
	cudaCheckError( __LINE__, cudaMemcpy(&queue_length,d_queue_length,sizeof(unsigned int), cudaMemcpyDeviceToHost));		

	while(queue_length !=0 ) {
		// unorder+thread queue + multiple dynamic parallelism
		/* Dynamic kernel configuration */
		if (queue_length<=maxDegreeT){
			dimGridT.x = 1;
		}
		else if (queue_length<=maxDegreeT*MAXDIMGRID){
			dimGridT.x = queue_length/maxDegreeT+1;
		}
		else{
			printf("Too many elements in queue\n");
			exit(0);	
		}		
		unorder_threadQueue_multiple_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, noNodeTotal,
																		d_work_queue, d_queue_length);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)));
		unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, noNodeTotal, d_work_queue, 
															d_queue_length, queue_max_length);
		cudaCheckError( __LINE__, cudaMemcpy(&queue_length,d_queue_length,sizeof(unsigned int), cudaMemcpyDeviceToHost));
		iteration++;
	}
	cudaCheckError( __LINE__, cudaFree(d_work_queue) );
	cudaCheckError( __LINE__, cudaFree(d_queue_length) );
}

void sssp_np_consolidate_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE) );

	unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, noNodeTotal, d_work_queue,
														d_queue_length, queue_max_length);
	cudaCheckError( __LINE__, cudaMemcpy(&queue_length,d_queue_length,sizeof(unsigned int), cudaMemcpyDeviceToHost));


	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bSize, sizeof(unsigned int) ) );
	unsigned int *d_count;
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );
	while(queue_length !=0 ) {
		//single dynamic parallelism
		/* Dynamic kernel configuration */
		if (queue_length<=maxDegreeT) {
			dimGridT.x = 1;
		}
		else if (queue_length<=maxDegreeT*MAXDIMGRID) {
			dimGridT.x = queue_length/maxDegreeT+1;
		}
		else{
			printf("Too many elements in queue\n");
			exit(0);
		}

		switch (config.solution) {
		case 1:
			consolidate_warp_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray,
																d_weightArray, d_update, noNodeTotal,
																d_work_queue, d_queue_length, d_buffer);
			break;
		case 2:
			consolidate_block_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray,
																d_weightArray, d_update, noNodeTotal,
																d_work_queue, d_queue_length, d_buffer);
			break;
		case 3:
			cudaCheckError( __LINE__, cudaMemset(d_bSize, 0, sizeof(unsigned int)));
			cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
			//int *test_buffer;
			//cudaCheckError( __LINE__, cudaMalloc( (void**)&test_buffer, sizeof(int*)) );
			//single_malloc<<<1,1>>>(test_buffer);
			consolidate_grid_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray,
																d_weightArray, d_update, noNodeTotal,
																d_work_queue, d_queue_length, d_buffer,
																d_bSize, d_count);

			//single_free<<<1,1>>>(test_buffer);
			/*
			cons_grid_dp_complex_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray,
																d_weightArray, d_update, noNodeTotal,
																d_work_queue, d_queue_length, d_buffer,
																d_bSize, d_count);
			*/
		default:
			printf("Unsupported solutions");
			exit(0);
		}	
		cudaCheckError( __LINE__, cudaGetLastError());
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)));
		unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, noNodeTotal, d_work_queue,
															d_queue_length, queue_max_length);

		cudaCheckError( __LINE__, cudaGetLastError());

		cudaCheckError( __LINE__, cudaMemcpy(&queue_length,d_queue_length,sizeof(unsigned int), cudaMemcpyDeviceToHost));
		iteration++;
	}
	cudaCheckError( __LINE__, cudaFree(d_work_queue) );
	cudaCheckError( __LINE__, cudaFree(d_queue_length) );
	cudaCheckError( __LINE__, cudaFree(d_buffer) );
}

void SSSP_GPU( ) 
{
	prepare_gpu();
	iteration = 0;
#ifdef GPU_PROFILE
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif

	start_time = gettime();
	switch (config.solution) {
		case 0:	sssp_tqueue_gpu();
			break;
		case 1: sssp_np_naive_gpu();
			break;
		case 2:	
		case 3:
		case 4: sssp_np_consolidate_gpu();
			break;
		default: 
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
	ker_exe_time += end_time - start_time;
	
	start_time = gettime();	
	cudaCheckError( __LINE__, cudaMemcpy( graph.costArray, d_costArray, sizeof(int)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime();
	d2h_memcpy_time += end_time - start_time;
	//fprintf(stderr, "SSSP iteration:\t\t%lf\n",end_time-start_time);
#ifdef GPU_PROFILE
	gpu_statistics<<<1,1>>>(config.solution);
#endif
	if (DEBUG)
		fprintf(stderr, "Iteration: %d\n", iteration);
	clean_gpu();
}
