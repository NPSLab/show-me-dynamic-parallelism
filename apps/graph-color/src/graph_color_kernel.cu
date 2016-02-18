#ifndef __GRAPH_COLOR_KERNEL__
#define __GRAPH_COLOR_KERNEL__

#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THREASHOLD 64
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 64
#define MAX_STREAM_NUM 4

//#define GPU_PROFILE

#ifdef GPU_PROFILE
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution) {
	printf("====>GPU #%u -number of kernel calls:%u\n", solution, nested_calls);
}

__global__ void reset_gpu_statistics() {
	nested_calls = 0;
}
#endif

__device__ unsigned int gm_idx_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];

__global__ void gclr_bitmap_kernel(	int *vertexArray, int *edgeArray, int *color,
		 	 	 	 	 	 	 	unsigned int *nonstop, int color_type, int nodeNumber )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool flag = true;
	if ( tid<nodeNumber && color[tid]==0 ) {
		/* get neighbor range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		/* access neighbors */
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			if ( (color[nid]==0 || color[nid]==color_type) &&  nid<tid ) { // nid is not colored and have smaller id
				flag = false;	break;
			}
		}
		if (flag) {
			color[tid] = color_type;
			*nonstop = 1;
		}
	}
}

__global__ void gclr_thread_queue_kernel(int *vertexArray, int *edgeArray, int *color,
										 int color_type, int nodeNumber,
										 int *queue, unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *qLength;

	bool flag = true;
	if ( tid<frontierNo ){
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbor range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbors */
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			if ( (color[nid]==0 || color[nid]==color_type) &&  nid<curr ) { // nid is not colored and have smaller id
				flag = false;	break;
			}
		}
		if (flag)  color[curr] = color_type;
	}
}

__global__ void gclr_block_queue_kernel(int *vertexArray, int *edgeArray, int *color,
										int color_type, int nodeNumber,
										int *queue, unsigned int *qLength)
{
	int bid = blockIdx.x+blockIdx.y*gridDim.x;	//*MAX_THREAD_PER_BLOCK + threadIdx.x;
	int frontierNo = *qLength;
	__shared__ bool flag;
	for ( ; bid<frontierNo; bid += gridDim.x * gridDim.y ){
		if (threadIdx.x==0) flag = true;
		__syncthreads();
		int curr = queue[bid];	//	grab a work from queue, bid is queue index
		/* get neighbor range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
			int nid = edgeArray[i];
			if ((color[nid]==0 || color[nid]==color_type) && nid<curr) {
				flag = false;	break;
			}
		}
		__syncthreads();
		if (threadIdx.x==0 && flag) {
			color[curr] = color_type;
		}
	}
}

/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */
/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void bitmap_process_neighbors(int curr, int *edgeArray, int *color,
								 		 int color_type, int start, int end,
										 unsigned int *nonstop)
{
	__shared__ bool flag ;
	if (threadIdx.x==0) flag = true;
	__syncthreads();
	for(int tid=threadIdx.x+start; tid<end; tid += blockDim.x){
		int nid = edgeArray[tid];
		if ( (color[nid]==0 || color[nid]==color_type) && nid<curr ) {
			flag = false; break;
		}
	}
	if (threadIdx.x==0 && flag) {
		*nonstop = 1;
		color[curr] = color_type;
	}
}

__global__ void process_neighbors(int curr, int *edgeArray, int *color,
								  int color_type, int start, int end)
{
	__shared__ bool flag ;
	if (threadIdx.x==0) flag = true;
	__syncthreads();
	for(int tid=threadIdx.x+start; tid<end; tid += blockDim.x){
		int nid = edgeArray[tid];
		if ( (color[nid]==0 || color[nid]==color_type) && nid<curr ) {
			flag = false; break;
		}
	}
	__syncthreads();
	if (threadIdx.x==0 && flag) color[curr] = color_type;
}

__global__ void gclr_bitmap_multidp_kernel(int *vertexArray, int *edgeArray, int *color,
											int color_type, int nodeNumber)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
 	
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
		//cudaStreamCreateWithFlags(&s[i], cudaStreamDefault);
	}
	bool flag = true;
	if ( tid<nodeNumber && color[tid]==0 ){
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {		
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( (color[nid]==0 || color[nid]==color_type) && nid<tid ){	// neighbour's level can be reduced
					flag = false; break;
				}
			}
			if (flag) color[tid] = color_type;
		}
		else {
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
			//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
     		process_neighbors<<<1, NESTED_BLOCK_SIZE,0, s[threadIdx.x%MAX_STREAM_NUM]>>>(tid, edgeArray, color, color_type, start, end);
		}
	}
}

/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void process_buffer( int *vertexArray, int *edgeArray, int *color, int color_type,
								int nodeNumber, int *buffer, unsigned int buffer_size)
{
	int bid = blockIdx.x;
	__shared__ bool flag;
	for ( ; bid<buffer_size; bid += gridDim.x ) {   // block-based mapping
		if (threadIdx.x==0) flag=true;
		__syncthreads();
		int curr = buffer[bid]; //nodes processed by current block
		/* get neighbor range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbors */
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
			int nid = edgeArray[eid]; // neighbour id
			if ( (color[nid]==0 || color[nid]==color_type) && nid<curr ) {
				flag = false; break;
			}
		}
		__syncthreads();
		if (threadIdx.x==0 && flag) color[curr] = color_type;
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void gclr_bitmap_cons_warp_dp_kernel(int *vertexArray, int *edgeArray, int *color, int color_type,
											int nodeNumber,	int *buffer)
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

	unsigned int *warp_index = &gm_idx_pool[blockIdx.x * warpDim + warpId];				// index of each block within its sub-buffer
	int t_idx = 0;						// used to access the buffer
	*warp_index = 0;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber && color[tid]==0 ){
		bool flag = true;
		/* get neighbor range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbors */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( (color[nid]==0 || color[nid]==color_type) && nid<tid ){	// neighbour's level can be reduced
					flag = false; break;
				}
			}
			if (flag) color[tid] = color_type;
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
		process_buffer<<<block_num,NESTED_BLOCK_SIZE,0, s[warpId%MAX_STREAM_NUM]>>>( vertexArray, edgeArray, color, color_type,
															nodeNumber, buffer+warp_offset, *warp_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void gclr_bitmap_cons_block_dp_kernel(int *vertexArray, int *edgeArray, int *color, int color_type,
											int nodeNumber,	int *buffer)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
    unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    unsigned int *block_index = &gm_idx_pool[blockIdx.x];	// index of each block within its sub-buffer
    int t_idx = 0;                                          // used to access the buffer
    if (threadIdx.x == 0) *block_index = 0;
    __syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber && color[tid]==0 ){
		bool flag = true;
		/* get neighbor range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbors */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( (color[nid]==0 || color[nid]==color_type) && nid<tid ){	// neighbour's level can be reduced
					flag = false; break;
				}
			}
			if (flag) color[tid] = color_type;
		}
		else {
			t_idx = atomicInc(block_index, per_block_buffer);
			buffer[t_idx+block_offset] = tid;
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
		process_buffer<<<block_num,NESTED_BLOCK_SIZE,0,s>>>( vertexArray, edgeArray, color, color_type,
															nodeNumber, buffer+block_offset, *block_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void gclr_bitmap_cons_grid_dp_kernel(int *vertexArray, int *edgeArray, int *color, int color_type,
												int nodeNumber, int *buffer, unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
    int t_idx = 0;                                          // used to access the buffer

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber && color[tid]==0 ){
		bool flag = true;
		/* get neighbor range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbors */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( (color[nid]==0 || color[nid]==color_type) && nid<tid ){	// neighbour's level can be reduced
					flag = false; break;
				}
			}
			if (flag) color[tid] = color_type;
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
			dim3 dimGridB(1,1,1);
			dimGridB.x = 13 * 16;
			if ( *idx<dimGridB.x ) dimGridB.x = *idx;
			gclr_block_queue_kernel<<<dimGridB, NESTED_BLOCK_SIZE>>>(vertexArray, edgeArray, color,
																	color_type, nodeNumber,buffer, idx);
		}
	}
}

__global__ void gen_queue_workset_kernel(	int *color, int *queue, unsigned int *queue_length,
											unsigned int queue_max_length, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && color[tid]==0 ) {
		/* write node number to queue */
		unsigned int q_idx = atomicInc(queue_length, queue_max_length);
		queue[q_idx] = tid;
	}
}

__global__ void check_workset_kernel(int *color, unsigned int *nonstop, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && color[tid]==0 ) {
		*nonstop = 1;
	}
}


#endif
