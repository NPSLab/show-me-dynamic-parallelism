#ifndef __SSSP_KERNEL__
#define __SSSP_KERNEL__

#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THRESHOLD 64
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 64
#define MAX_STREAM_NUM 4

//#define GPU_PROFILE

#ifdef GPU_PROFILE

__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
        printf("====> GPU #%u - number of nested kernel calls:%u\n",solution, nested_calls);
}

__global__ void reset_gpu_statistics() {
	nested_calls = 0;
}
#endif

__device__ unsigned int gm_idx_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];
__device__ int *gm_buffer_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];

__global__ void single_malloc(int *buffer)
{
	buffer = (int*)malloc(sizeof(unsigned int)*GM_BUFF_SIZE);
}

__global__ void single_free(int *buffer)
{
	free(buffer);
}

__global__ void unorder_threadQueue_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue,unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *qLength;
	if ( tid<frontierNo ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			int alt = costCurr + weightArray[i];
			if ( costArray[nid] > alt ) {
				atomicMin(costArray+nid, alt);
				update[nid] = 1;	// update neighbour needed
			}
		}
	}
}

__global__ void unorder_blockQueue_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue, unsigned int *qLength)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x; //*MAX_THREAD_PER_BLOCK + threadIdx.x;	
	int frontierNo = *qLength;
	for ( ; bid<frontierNo; bid += gridDim.x * gridDim.y ) {
		int curr = queue[bid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int eid=threadIdx.x+start; eid<end; eid += blockDim.x) {
			int nid = edgeArray[eid];	// neighbour id
			int alt = costCurr + weightArray[eid];
			if ( costArray[nid] > alt ) {
				atomicMin(costArray+nid, alt);				
				update[nid] = 1;	// update neighbour needed
			}
		}
	}
}


/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void sssp_process_buffer( int *vertexArray, int *edgeArray, int *weightArray, int *costArray, 
				     char *update, int nodeNumber, int *buffer, unsigned int buffer_length)
{
	int bid = blockIdx.x; 
	for (; bid<buffer_length; bid += gridDim.x ) {   // block-based mapping
		int curr = buffer[bid];	//nodes processed by current block
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x){ // eid is the identifier of the edge processed by the current thread
			if ( eid<end ){
				int nid = edgeArray[eid];	// neighbour id
				int alt = costCurr + weightArray[eid];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);				
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
	}
}

/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */

/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void sssp_process_neighbors(	int *edgeArray, int *weightArray, int *costArray, 
					char *update, int costCurr, int start, int end)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
	for (; tid < end; tid += gridDim.x * blockDim.x ) {
       	int nid = edgeArray[tid];
		int alt = costCurr + weightArray[tid];
		if ( costArray[nid] > alt ) {
			atomicMin(costArray+nid, alt);
			update[nid] = 1;	// update neighbour needed
		}
	}
}



/* thread queue with dynamic parallelism and potentially multiple nested kernel calls */
__global__ void unorder_threadQueue_multiple_dp_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
							char *update, int nodeNumber, int *queue,unsigned int *queue_length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *queue_length;
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
		//cudaStreamCreateWithFlags(&s[i], cudaStreamDefault);
	}
	if ( tid<frontierNo ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edgeNum = end - start;
		if ( edgeNum<THRESHOLD ) {
			/* access neighbours */
			int costCurr = costArray[curr];
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				int alt = costCurr + weightArray[i];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
		else {
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
		//	printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
		int costCurr = costArray[curr];

		//sssp_process_neighbors<<<edgeNum/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE, 0, s[threadIdx.x%MAX_STREAM_NUM] >>>(
		//			 	 edgeArray, weightArray, costArray, update, costCurr, start, end);
		//int num_block = edgeNum/NESTED_BLOCK_SIZE < 2 ? edgeNum/NESTED_BLOCK_SIZE : 2;
		sssp_process_neighbors<<<1, NESTED_BLOCK_SIZE, 0, s[threadIdx.x%MAX_STREAM_NUM] >>>(
					 	 edgeArray, weightArray, costArray, update, costCurr, start, end);

		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-warp */
__global__ void consolidate_warp_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer)
{
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
	}
	int warpId = threadIdx.x / WARP_SIZE;
	int warpDim = blockDim.x / WARP_SIZE;
	int total_warp_num = gridDim.x * warpDim;
	//unsigned warp_buffer_size = GM_BUFF_SIZE/total_warp_num; 	// amount of the buffer available to each warp
	unsigned warp_buffer_size = 256; 	// amount of the buffer available to each warp
	int **warp_buffer = &gm_buffer_pool[blockIdx.x * warpDim + warpId];	 // index of each block within its sub-buffer
	unsigned int *warp_index = &gm_idx_pool[blockIdx.x * warpDim + warpId];	 // index of each block within its sub-buffer
#if BUFFER_ALLOCATOR == 0  // default
	if (threadIdx.x%WARP_SIZE==0) {
		*warp_buffer = (int*)malloc(sizeof(unsigned int)*warp_buffer_size);
	}

#elif BUFFER_ALLOCATOR == 1  // halloc
	if (threadIdx.x%WARP_SIZE==0) {
		*warp_buffer = (int*)hamalloc(sizeof(unsigned int)*warp_buffer_size);
	}
#else  // customized allocator
	unsigned warp_offset = (blockIdx.x * warpDim + warpId) * warp_buffer_size;  // block offset within the buffer
	*warp_buffer = &buffer[warp_offset];
#endif
	int t_idx = 0;						// used to access the buffer
	*warp_index = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  // grab a work from queue, tid is queue index
        /* get neighbor range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbors */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
						update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(warp_index, warp_buffer_size);
			(*warp_buffer)[t_idx] = queue[tid];
        }
	}

    //2nd phase - nested kernel call
	if (threadIdx.x%WARP_SIZE==0 && *warp_index!=0){
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		unsigned int block_num = 13;
		if ( *warp_index<block_num ) block_num = *warp_index; 
      	sssp_process_buffer<<<block_num,NESTED_BLOCK_SIZE,0, s[warpId%MAX_STREAM_NUM]>>>( vertexArray, edgeArray, weightArray, costArray,
        						  		update, nodeNumber, *warp_buffer, *warp_index);
	}
	if (threadIdx.x%WARP_SIZE==0) {
#if BUFFER_ALLOCATOR == 0  // default
		cudaDeviceSynchronize();
		free(*warp_buffer);
#elif BUFFER_ALLOCATOR == 1 // halloc
		cudaDeviceSynchronize();
		hafree(*warp_buffer);
#else

#endif
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void consolidate_block_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	//unsigned block_buffer_size = GM_BUFF_SIZE/gridDim.x; 	// amount of the buffer available to each thread block
	unsigned block_buffer_size = 2048;	// amount of the buffer available to each thread block
	unsigned block_offset = blockIdx.x * block_buffer_size;  // block offset within the buffer
   __shared__ int *block_buffer;
	unsigned int *block_index = &gm_idx_pool[blockIdx.x];	// index of each block within its sub-buffer

#if BUFFER_ALLOCATOR == 0  // default
	if (threadIdx.x==0) {
		block_buffer = (int*)malloc(sizeof(unsigned int)*block_buffer_size);
	}
	//block_buffer = &buffer[block_offset];
#elif BUFFER_ALLOCATOR == 1 // halloc
	if (threadIdx.x==0) {
		block_buffer = (int*)hamalloc(sizeof(unsigned int)*block_buffer_size);
	}
#else	// customized allocator
	if (threadIdx.x==0) {
		block_buffer = &buffer[block_offset];
	}
#endif
	int t_idx = 0;						// used to access the buffer
	if (threadIdx.x == 0) *block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbours */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(block_index, block_buffer_size);
            //buffer[t_idx+block_offset] = queue[tid];
        	block_buffer[t_idx] = queue[tid];
        }
	}
	__syncthreads();
    //2nd phase - nested kernel call
	if (threadIdx.x==0 && *block_index!=0){
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		unsigned int block_num = 13;
		if ( *block_index<block_num ) block_num = *block_index;
      	sssp_process_buffer<<<block_num,NESTED_BLOCK_SIZE,0,s>>>( vertexArray, edgeArray, weightArray, costArray,
        						  								update, nodeNumber, block_buffer, *block_index);
  
#if BUFFER_ALLOCATOR == 0  // default
		cudaDeviceSynchronize();
		free(block_buffer);
#elif BUFFER_ALLOCATOR == 1  // default
		cudaDeviceSynchronize();
		hafree(block_buffer);
#endif
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void consolidate_grid_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                            char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                            int *buffer, unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	int t_idx = 0;						// used to access the buffer
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbours */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(idx, GM_BUFF_SIZE);
        	buffer[t_idx] = queue[tid];
        }
	}
	__syncthreads();
	// 2nd phase, grid level consolidation
	if (threadIdx.x==0) {
		// count up
		if ( atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) ) {//
			//printf("gridDim.x: %d buffer: %d\n", gridDim.x, *idx);
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
#endif
			dim3 dimGridB(1,1,1);
			dimGridB.x = 13 * 16;
			if ( *idx < dimGridB.x ) dimGridB.x = *idx; 
			unorder_blockQueue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(	vertexArray, edgeArray, costArray,
																		weightArray, update, nodeNumber,
																		buffer, idx);
		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per grid 
 * shared memory is used for block-level buffer but no benefit for performance
 */
__global__ void cons_grid_dp_complex_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer, unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x; 	// amount of the buffer available to each thread block
	//unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    __shared__ int shm_buffer[MAXDIMBLOCK];
	unsigned int *block_index = &gm_idx_pool[blockIdx.x];				// index of each block within its sub-buffer
	__shared__ int offset;
	int t_idx = 0;						// used to access the buffer
	if (threadIdx.x == 0) *block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbours */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(block_index, per_block_buffer);
            //buffer[t_idx+block_offset] = queue[tid];
        	shm_buffer[t_idx] = queue[tid];
        }
	}
	__syncthreads();
	// reorganize consolidation buffer for load balance (get offset per block)
	if (threadIdx.x==0) {
		offset = atomicAdd(idx, *block_index);
		//printf("blockIdx.x: %d block idx: %d idx: %d\n", blockIdx.x, block_index, offset);
	}
	__syncthreads();
	// dump shm_buffer to global buffer
	if (threadIdx.x<*block_index) {
		int gm_idx = threadIdx.x + offset;
		buffer[gm_idx] = shm_buffer[threadIdx.x];
	}
	__syncthreads();
	// 2nd phase, grid level consolidation
	if (threadIdx.x==0) {
		// count up
		if ( atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) ) {//
			//printf("gridDim.x: %d buffer: %d\n", gridDim.x, *idx);
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
#endif
			dim3 dimGridB(1,1,1);
			if (*idx<=MAXDIMGRID) {
				dimGridB.x = *idx;
			}
			else if (*idx<=MAXDIMGRID*NESTED_BLOCK_SIZE) {
				dimGridB.x = MAXDIMGRID;
				dimGridB.y = *idx/MAXDIMGRID+1;
			}
			else {
				printf("Too many elements in queue\n");
			}
			unorder_blockQueue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(	vertexArray, edgeArray, costArray,
																		weightArray, update, nodeNumber,
																		buffer, idx);
		}
	}
}

/* LOAD BALANCING BY USING MULTIPLE QUEUES */
/* divides the nodes into two queues */ 
__global__ void unorder_gen_multiQueue_kernel(	int *vertexArray, char *update, int nodeNumber, 
						int *queue_l, unsigned int *qCounter_l, unsigned int qMaxLength_l,
						int *queue_h, unsigned int *qCounter_h, unsigned int qMaxLength_h)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edgeNum = end - start;
		if ( edgeNum<THRESHOLD ) {
			/* write vertex number to LOW degree queue */
			unsigned int qIndex = atomicInc(qCounter_l, qMaxLength_l);
			queue_l[qIndex] = tid;
		}
		else {
			/* write vertex number to HIGH degree queue */
			unsigned int qIndex = atomicInc(qCounter_h, qMaxLength_h);
			queue_h[qIndex] = tid;
		}
	}
}
__global__ void unorder_generateQueue_kernel(	char *update, int nodeNumber, int *queue, 
												unsigned int *qCounter, unsigned int qMaxLength)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		/* write node number to queue */
		unsigned int qIndex = atomicInc(qCounter, qMaxLength);
		queue[qIndex] = tid;
	}
}
#endif
