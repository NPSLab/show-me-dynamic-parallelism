#ifndef __PG_KERNEL__
#define __PG_KERNEL__

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THREASHOLD 64
#define SHM_BUFF_SIZE 256

#define NESTED_BLOCK_SIZE 64
#define WARP_SIZE 32
#define MAX_STREAM_NUM 4

#define GPU_PROFILE

#ifdef GPU_PROFILE
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution) {
	printf("====>GPU #%u - number of kernel calls:%u\n", solution, nested_calls);
}

__global__ void reset_gpu_statistics() {
	nested_calls = 0;
}
#endif

__device__ unsigned int gm_idx_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];

#if CUDA_VERSION <= 6000
__device__ inline double __shfl_down (double var, unsigned int src_lane, int width=32)
{
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_down(a.x, src_lane, width);
	a.y = __shfl_down(a.y, src_lane, width);
	return *reinterpret_cast<double*>(&a);
}
#endif

__inline__ __device__ double warp_reduce_sum(double val) {
	for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
		val += __shfl_down(val, offset);
	return val;
}

__inline__ __device__ double block_reduce_sum(double val) {
	static __shared__ double shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	val = warp_reduce_sum(val);     // Each warp performs partial reduction
	if (lane==0) shared[wid]=val;	// Write reduced value to shared memory
	__syncthreads();              // Wait for all partial reductions
	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
	if (wid==0) val = warp_reduce_sum(val); //Final reduce within first warp
	return val;
}

__device__ double atomicAdd(double* address, double val) 
{ 
	unsigned long long int* address_as_ull = (unsigned long long int*)address; 
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __longlong_as_double(old); 
}

__global__ void check_delta_kernel(FLOAT_T *rankArray, FLOAT_T *newRankArray, unsigned int *nonstop, int node_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<node_num ) {
		FLOAT_T ep = fabs(rankArray[tid]-newRankArray[tid]);
		if ( ep>EPSILON ) {
			*nonstop = 1;
		}
	}
}


__global__ void update_danglingrankarray_kernel(FLOAT_T *rankArray, FLOAT_T *danglingRankArray, int *danglingVertexArray, int noDanglingNode)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<noDanglingNode ) {
		int index = danglingVertexArray[tid];
		danglingRankArray[tid] = rankArray[index];
	}

}

/* pull method */
__global__  void pagerank_thread_pull_kernel(	int *child_vertex_array, int *r_edge_array, int *outdegree_array,
												FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
												FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_T new_rank_tid = 0.0;
    if ( tid<node_num ) { 
		int start = child_vertex_array[tid];
        int end = child_vertex_array[tid+1];
		/*for (int i=start; i<end; ++i) {
 			int parent = r_edge_array[i];
			FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
			new_rank_tid += damping * rank;
		}
		new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        new_rank_array[tid] = new_rank_tid;*/
        for (int i=start; i<end; ++i) {
         	int parent = r_edge_array[i];
        	FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
        	new_rank_tid += damping * rank;
       	}
       	new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
       	new_rank_array[tid] = new_rank_tid;
   	}   
}

__global__ void pg_thread_queue_kernel(	int *child_vertex_array, int *r_edge_array, int *outdegree_array, 
										FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
										FLOAT_T rank_dangling_node, FLOAT_T damping,	int node_num,
										int *queue, unsigned int *queue_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontier_no = *queue_size;
	FLOAT_T new_rank_tid = 0.0;
	int curr = 0;
	//printf("Thread %d process node %d\n", tid, curr);
	if ( tid<frontier_no ){
		curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */
		int start = child_vertex_array[curr];
		int end = child_vertex_array[curr+1];
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			int parent = r_edge_array[i];
			FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
			new_rank_tid += damping * rank;
		}
		new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        new_rank_array[curr] = new_rank_tid;
	}
}

__global__ void pg_block_queue_kernel(	int *child_vertex_array, int *r_edge_array, int *outdegree_array, 
										FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
										FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num,
										int *queue, unsigned int *queue_size )
{
	int bid = blockIdx.x+blockIdx.y*gridDim.x;	//*MAX_THREAD_PER_BLOCK + threadIdx.x;
	int frontier_no = *queue_size;
	for ( ; bid<frontier_no; bid += gridDim.x * gridDim.y ) {
		FLOAT_T new_rank_tid = 0.0;
		int curr = queue[bid];	//	grab a work from queue, bid is queue index
		/* get neighbour range */
		int start = child_vertex_array[curr];
		int end = child_vertex_array[curr+1];
		/* access neighbours */
		for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
			int parent = r_edge_array[i];
			FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
			new_rank_tid += damping * rank;
		}
		//s_new_rank_tid = atomicAdd( &s_new_rank_tid, new_rank_tid);
		__syncthreads();
		new_rank_tid = block_reduce_sum( new_rank_tid );
		//new_rank_tid = 0.5;
		if ( threadIdx.x==0 ) {
			//new_rank_tid = s_new_rank_tid;
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[curr] = new_rank_tid;
		}
	}
}



/* LOAD BALANCING THROUGH DELAYED BUFFER */

/* implements a delayed buffer in shared memory: 
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread) 
   - in phase 2, the blocks access the nodes in the delayed-buffer in a block-based mapping (one neighbor per thread) 
*/
__global__  void pg_shared_delayed_buffer_kernel(	int *child_vertex_array, int *r_edge_array, int *outdegree_array,
												FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
												FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_T new_rank_tid = 0.0;
	
	int t_idx = 0;
	__shared__ int buffer[SHM_BUFF_SIZE];
	__shared__ unsigned int idx;
	if ( threadIdx.x==0 ) idx = 0;
	__syncthreads();

	// 1st phase - thread-based mapping
    if ( tid<node_num ) { 
		int start = child_vertex_array[tid];
        int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
   		}
		else {	// insert into delayed buffer
			t_idx = atomicInc( &idx, SHM_BUFF_SIZE);
			buffer[t_idx] = tid;
		} 
	}  
	__syncthreads();
	// 2nd phase - each block processed all the elements in its shared memory buffer; each thread process a different neighbor
#ifdef GPU_PROFILE
	if (tid==0 && idx!=0) {
		printf("In Block %d # delayed nodes : %d\n", blockIdx.x, idx);
	}
#endif
	for (int i=0; i<idx; i++) {
		int curr = buffer[i]; //grab an element from the buffer
		new_rank_tid = 0.0;
		// get neighbour range
		int start = child_vertex_array[curr];
		int end = child_vertex_array[curr+1];
		// access neighbors - one thread per neigbor;
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x){
			int parent = r_edge_array[eid]; // parent id
			FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
			new_rank_tid += damping * rank;
		}
		//s_new_rank_tid = atomicAdd( &s_new_rank_tid, new_rank_tid);
		//__syncthreads();
		__syncthreads();
		new_rank_tid = block_reduce_sum( new_rank_tid );
		if ( threadIdx.x==0 ) {
			//new_rank_tid = s_new_rank_tid;
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[curr] = new_rank_tid;
		}
	}
}

/* implements phase 1 of delayed buffer (buffer) in global memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - phase 2 must be implemented by separately invoking the "process_buffer" kernel
*/
__global__ void pg_global_delayed_buffer_kernel(int *child_vertex_array, int *r_edge_array, int *outdegree_array,
												FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
												FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num,
												int *buffer, unsigned int *idx )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_T new_rank_tid = 0.0;
	int t_idx = 0;

	// 1st phase - thread-based mapping
    if ( tid<node_num ) { 
		int start = child_vertex_array[tid];
        int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
   		}
		else {	// insert into delayed buffer
			t_idx = atomicInc( idx, GM_BUFF_SIZE);
			buffer[t_idx] = tid;
		} 
	}  
}


/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */
/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void pg_process_neighbors(	int *r_edge_array, int *outdegree_array, FLOAT_T *rank_array,
										FLOAT_T *new_rank_array, FLOAT_T rank_random_walk, FLOAT_T rank_dangling_node,
										FLOAT_T damping, int start, int end, int nid)
{
	int tid = threadIdx.x + start;
	FLOAT_T new_rank_tid = 0.0;
	for ( ; tid<end; tid += blockDim.x ) {
		int parent = r_edge_array[tid];
		FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
		new_rank_tid += damping * rank;
	}
	__syncthreads();
	new_rank_tid = block_reduce_sum( new_rank_tid );
	if ( threadIdx.x==0 ) {
		new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node; 
		new_rank_array[nid] = new_rank_tid;
	}
}

__global__  void pg_multidp_kernel(	int *child_vertex_array, int *r_edge_array, int *outdegree_array,
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
	}   

    FLOAT_T new_rank_tid = 0.0;
	// 1st phase - thread-based mapping
    if ( tid<node_num ) { 
		int start = child_vertex_array[tid];
        int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
   		}
		else {
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
			//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
			//if (tid==18929 || tid==18867)
			//printf("Node %d: old %f  new %f  \n", tid, rank_array[tid], new_rank_array[tid]);
			new_rank_array[tid] = 0.0;
     		pg_process_neighbors<<<1, NESTED_BLOCK_SIZE, 0, s[threadIdx.x%MAX_STREAM_NUM]>>>(
							r_edge_array, outdegree_array, rank_array, new_rank_array, 
							rank_random_walk, rank_dangling_node, damping, start, end, tid);
			
			cudaDeviceSynchronize();
			//if (tid==18929 || tid==18867)
			//printf("Node %d with %d edges : old %f  new %f  \n", tid, end-start, rank_array[tid], new_rank_array[tid]);
		} 
	}  
}

/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void pg_process_buffer(	int *child_vertex_array, int *r_edge_array, int *outdegree_array, 
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num, int *buffer,
									unsigned int buffer_size)
{
	int bid = blockIdx.x;
	for ( ; bid<buffer_size; bid += gridDim.x ) {   // block-based mapping
		FLOAT_T new_rank_tid = 0.0;
		int curr = buffer[bid]; //nodes processed by current block
		/* get neighbour range */
		int start = child_vertex_array[curr];
		int end = child_vertex_array[curr+1];
		/* access neighbours */
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
			int parent = r_edge_array[eid]; // neighbour id
			FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
			new_rank_tid += damping * rank;
		}
		__syncthreads();
		new_rank_tid = block_reduce_sum( new_rank_tid );
		if ( threadIdx.x==0 ) {
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[curr] = new_rank_tid;
		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void pg_singledp_kernel(int *child_vertex_array, int *r_edge_array, int *outdegree_array,
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num,
									int *buffer )
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
    unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    unsigned int *block_index = &gm_idx_pool[blockIdx.x];                            // index of each block within its sub-buffer
    int t_idx = 0;                                          // used to access the buffer
    if (threadIdx.x == 0) *block_index = 0;
    __syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_T new_rank_tid = 0.0;

	// 1st phase
	if ( tid<node_num ) {
		/* get neighbour range */
		int start = child_vertex_array[tid];
		int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
   			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
		}
		else {
			t_idx = atomicInc(block_index, per_block_buffer);
			buffer[t_idx+block_offset] = tid;
		}
	}
	__syncthreads();
	if (threadIdx.x==0 && *block_index!=0){
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		// each node is processed by single block, blockIdx.x == block_index
		pg_process_buffer<<<*block_index,NESTED_BLOCK_SIZE,0,s>>>(	child_vertex_array, r_edge_array, outdegree_array,
																rank_array, new_rank_array, rank_random_walk,
																rank_dangling_node,	damping, node_num, 
																buffer+block_offset, *block_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void pg_warp_dp_kernel(int *child_vertex_array, int *r_edge_array, int *outdegree_array,
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num,
									int *buffer )
{
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
	}
	int warpId = threadIdx.x / WARP_SIZE;
	int warpDim = blockDim.x / WARP_SIZE;
	int total_warp_num = gridDim.x * warpDim;
	unsigned per_warp_buffer = GM_BUFF_SIZE/total_warp_num; 	// amount of the buffer available to each thread block
	unsigned warp_offset = (blockIdx.x * warpDim + warpId) * per_warp_buffer;  // block offset within the buffer
	unsigned int *warp_index = &gm_idx_pool[blockIdx.x * warpDim + warpId];  	// index of each block within its sub-buffer
	int t_idx = 0;						// used to access the buffer
	*warp_index = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_T new_rank_tid = 0.0;

	// 1st phase
	if ( tid<node_num ) {
		/* get neighbour range */
		int start = child_vertex_array[tid];
		int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
   			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
		}
		else {
			t_idx = atomicInc(warp_index, per_warp_buffer);
			buffer[t_idx+warp_offset] = tid;
		}
	}

	//2nd phase - nested kernel call
	if (threadIdx.x%WARP_SIZE==0 && *warp_index!=0){
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		unsigned int block_num = 13;
		if ( *warp_index<block_num ) block_num = *warp_index;
	    pg_process_buffer<<<block_num, NESTED_BLOCK_SIZE,0,s[threadIdx.x%MAX_STREAM_NUM]>>>(	child_vertex_array, r_edge_array, outdegree_array,
	      																	rank_array, new_rank_array, rank_random_walk,
	      																	rank_dangling_node,	damping, node_num,
	      																	buffer+warp_offset, *warp_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void pg_block_dp_kernel(int *child_vertex_array, int *r_edge_array, int *outdegree_array,
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num,
									int *buffer )
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	//unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
	unsigned per_block_buffer = 2048;	     // amount of the buffer available to each thread block
    unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    unsigned int *block_index = &gm_idx_pool[blockIdx.x];                            // index of each block within its sub-buffer
    int t_idx = 0;                                          // used to access the buffer
    if (threadIdx.x == 0) *block_index = 0;
    __syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_T new_rank_tid = 0.0;

	// 1st phase
	if ( tid<node_num ) {
		/* get neighbour range */
		int start = child_vertex_array[tid];
		int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
   			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
		}
		else {
			t_idx = atomicInc(block_index, per_block_buffer);
			buffer[t_idx+block_offset] = tid;
		}
	}
	__syncthreads();
	if (threadIdx.x==0 && *block_index!=0){
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		// each node is processed by single block, blockIdx.x == block_index
		//unsigned int block_num = 13;
		//if ( *block_index<block_num ) block_num = *block_index;
		pg_process_buffer<<<*block_index,NESTED_BLOCK_SIZE,0,s>>>(	child_vertex_array, r_edge_array, outdegree_array,
																rank_array, new_rank_array, rank_random_walk,
																rank_dangling_node,	damping, node_num,
																buffer+block_offset, *block_index);
	}
}


/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void pg_grid_dp_kernel(int *child_vertex_array, int *r_edge_array, int *outdegree_array,
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num,
									int *buffer, unsigned int *idx, unsigned int *count )
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    int t_idx = 0;                                          // used to access the buffer

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_T new_rank_tid = 0.0;

	// 1st phase
	if ( tid<node_num ) {
		/* get neighbour range */
		int start = child_vertex_array[tid];
		int end = child_vertex_array[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
   			/* access neighbours */
			for (int i=start; i<end; ++i) {
 				int parent = r_edge_array[i];
				FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
				new_rank_tid += damping * rank;
			}
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[tid] = new_rank_tid;
		}
		else {
			t_idx = atomicInc(idx, GM_BUFF_SIZE);
			buffer[t_idx] = tid;
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
			//printf("buffer size: %d\n", *idx);
			dim3 dimGridB(208,1,1);
			if ( *idx<dimGridB.x ) dimGridB.x = *idx;
			pg_block_queue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(	child_vertex_array, r_edge_array, outdegree_array,
																		rank_array, new_rank_array, rank_random_walk,
																		rank_dangling_node, damping, node_num,
																		buffer, idx );
		}
	}
}

__global__ void gen_queue_workset_kernel(	char *update, int *queue, unsigned int *queue_length, 
											unsigned int queue_max_length, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		/* write node number to queue */
		unsigned int q_idx = atomicInc(queue_length, queue_max_length);
		queue[q_idx] = tid;
	}
}

__global__ void gen_dual_queue_workset_kernel(int *vertexArray, char *update, int nodeNumber,
								int *queue_l, unsigned int *queue_length_l, unsigned int queue_max_length_l,
								int *queue_h, unsigned int *queue_length_h, unsigned int queue_max_length_h)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* write vertex number to LOW degree queue */
			unsigned int q_idx = atomicInc(queue_length_l, queue_max_length_l);
			queue_l[q_idx] = tid;
		}
		else {
			/* write vertex number to HIGH degree queue */
			unsigned int q_idx = atomicInc(queue_length_h, queue_max_length_h);
			queue_h[q_idx] = tid;
		}
	}
}

#endif
