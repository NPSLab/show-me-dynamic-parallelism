#ifndef __UBFS_KERNEL__
#define __UBFS_KERNEL__

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THREASHOLD 64
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 64
#define MAX_STREAM_NUM 4

#define WARP_SIZE 32

//#define GPU_PROFILE

#ifdef GPU_PROFILE
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
	printf("====> GPU #%u - number of nested kernel calls:%u\n", solution, nested_calls);
}

__global__ void reset_gpu_statistics() {
	nested_calls = 0;
}
#endif

__device__ unsigned int gm_idx_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];
__device__ int *gm_buffer_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];

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

#if CUDA_VERSION <= 6000
__device__ inline double __shfl_down (double var, unsigned int src_lane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, src_lane, width);
    a.y = __shfl_down(a.y, src_lane, width);
    return *reinterpret_cast<double*>(&a);
}
#endif

__inline__ __device__ FLOAT_T warp_reduce_sum(FLOAT_T val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__ FLOAT_T block_reduce_sum(FLOAT_T val) {
    static __shared__ FLOAT_T shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);     // Each warp performs partial reduction
    if (lane==0) shared[wid]=val;   // Write reduced value to shared memory
    __syncthreads();              // Wait for all partial reductions
    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid==0) val = warp_reduce_sum(val); //Final reduce within first warp
    return val;
}

__global__ void	spmv_bitmap_init_kernel( char *frontier, int node_num )
{
	/* row == tid, each thread processes one row in sparse matrix */
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<node_num )
		frontier[tid] = 1;
}

__global__ void	csr_spmv_kernel( int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x, FLOAT_T *y, int node_num )
{
	/* row == tid, each thread processes one row in sparse matrix */
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( tid<node_num ) {
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			dot += data[i] * x[indices[i]];
		}
		y[tid] = dot;
	}
}

__global__ void	csr_spmv_thread_queue_kernel( int *ptr, int *indices, int *queue, unsigned int *queue_length, 
											FLOAT_T *data, FLOAT_T *x, FLOAT_T *y, int node_num )
{
	/* row == tid, each thread processes one row in sparse matrix */
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int frontier_size = *queue_length;
	if ( tid<frontier_size ) {
		int row = queue[tid];
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[row];
		int end = ptr[row+1];
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			dot += data[i] * x[indices[i]];
		}
		y[row] = dot;
	}
}

__global__ void	csr_spmv_block_queue_kernel(int *ptr, int *indices, int *queue, unsigned int *queue_length,
											FLOAT_T *data, FLOAT_T *x, FLOAT_T *y, int node_num )
{
	int bid = blockIdx.x+blockIdx.y*gridDim.x;	//*MAXDIMBLOCK + threadIdx.x;

	unsigned int frontier_size = *queue_length;
	for ( ; bid<frontier_size; bid += gridDim.x * gridDim.y ){
		int row = 0;
		FLOAT_T dot = 0;
		row = queue[bid];	//	grab a work from queue, bid is queue index
		/* get neighbour range */
		int start = ptr[row];
		int end = ptr[row+1];
		/* access neighbours */
		for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
			dot += data[i] * x[indices[i]];
		}
		__syncthreads();
		dot = block_reduce_sum( dot );
		if ( threadIdx.x==0 ) {
			y[row] = dot;
		}
	}
}

/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */
/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void spmv_process_neighbors( int *indices, FLOAT_T *data, FLOAT_T *x, FLOAT_T *y, 
										int start, int end, int row )
{	int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
	FLOAT_T dot = 0;
	for (; tid < end; tid += gridDim.x*blockDim.x ) {
		dot += data[tid] * x[indices[tid]];
	}
	__syncthreads();
	dot = block_reduce_sum( dot );
	if ( threadIdx.x==0 )
		atomicAdd( &y[row], dot);
}

__global__ void csr_spmv_multidp_kernel(int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x,
										FLOAT_T *y, int node_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
	}

	if ( tid<node_num ) {
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {		
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				dot += data[i] * x[indices[i]];
			}
			y[tid] = dot;
		}
		else {
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
			//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
     		//spmv_process_neighbors<<<edge_num/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE,0,s[threadIdx.x%MAX_STREAM_NUM]>>>(
 			//	  								         indices, data, x, y, start, end, tid);
     		spmv_process_neighbors<<<1, NESTED_BLOCK_SIZE,0,s[threadIdx.x%MAX_STREAM_NUM]>>>(
 				  								         indices, data, x, y, start, end, tid);
		}
	}
}

/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void spmv_process_buffer( int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x, FLOAT_T *y,
									int node_num, int *buffer, unsigned int buffer_size)
{
	int bid = blockIdx.x;
	for ( ; bid<buffer_size; bid += gridDim.x ) {   // block-based mapping
		FLOAT_T dot = 0;
		int row = buffer[bid]; //nodes processed by current block
		/* get neighbour range */
		int start = ptr[row];
		int end = ptr[row+1];
		/* access neighbours */
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
			dot += data[eid] * x[indices[eid]];	
		}
		__syncthreads();
		dot = block_reduce_sum( dot );
		if ( threadIdx.x==0 ) {
			y[row] = dot;
		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void csr_spmv_warp_dp_kernel(int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x,
										FLOAT_T *y, int node_num, int *buffer)
{
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
	}
	int warpId = threadIdx.x / WARP_SIZE;
	int warpDim = blockDim.x / WARP_SIZE;
	int total_warp_num = gridDim.x * warpDim;
	//unsigned warp_buffer_size = GM_BUFF_SIZE/total_warp_num; 	// amount of the buffer available to each thread block
	unsigned warp_buffer_size = 64; 	// amount of the buffer available to each thread block
	int **warp_buffer = &gm_buffer_pool[blockIdx.x * warpDim + warpId];
	unsigned int *warp_index = &gm_idx_pool[blockIdx.x * warpDim + warpId];		// index of each block within its sub-buffer
#if BUFFER_ALLOCATOR == 0 // default
	if (threadIdx.x%WARP_SIZE==0) {
		*warp_buffer = (int*)malloc(sizeof(unsigned int)*warp_buffer_size);
	}
#elif BUFFER_ALLOCATOR == 1 // halloc
	if (threadIdx.x%WARP_SIZE==0) {
		*warp_buffer = (int*)hamalloc(sizeof(unsigned int)*warp_buffer_size);
	}
#else
	unsigned warp_offset = (blockIdx.x * warpDim + warpId) * warp_buffer_size;  // block offset within the buffer
	*warp_buffer = &buffer[warp_offset];
#endif
	int t_idx = 0;						// used to access the buffer
	*warp_index = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<node_num){
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				dot += data[i] * x[indices[i]];
			}
			y[tid] = dot;
		}
		else {
			t_idx = atomicInc(warp_index, warp_buffer_size);
			(*warp_buffer)[t_idx] = tid;
		}
	}
	__syncthreads();
	//2nd phase - nested kernel call
	if (threadIdx.x%WARP_SIZE==0 && *warp_index!=0){
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		unsigned int block_num = 13;
		if ( *warp_index<block_num ) block_num = *warp_index;
		spmv_process_buffer<<<block_num,NESTED_BLOCK_SIZE,0,s[threadIdx.x%MAX_STREAM_NUM]>>>( ptr, indices, data, x, y, node_num,
																*warp_buffer, *warp_index);
	}
#if BUFFER_ALLOCATOR == 0 // default
	if (threadIdx.x%WARP_SIZE==0) {
		cudaDeviceSynchronize();
		free(*warp_buffer);
	}
#elif BUFFER_ALLOCATOR == 1 // halloc
	if (threadIdx.x%WARP_SIZE==0) {
		cudaDeviceSynchronize();
		hafree(*warp_buffer);
	}
#endif
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void csr_spmv_block_dp_kernel(int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x,
										FLOAT_T *y, int node_num, int *buffer)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	//unsigned block_buffer_size = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
	unsigned block_buffer_size = 2048;     // amount of the buffer available to each thread block
    unsigned block_offset = blockIdx.x * block_buffer_size;  // block offset within the buffer
    __shared__ int *block_buffer;
    unsigned int *block_index = &gm_idx_pool[blockIdx.x];                            // index of each block within its sub-buffer
#if BUFFER_ALLOCATOR == 0 // default
	if (threadIdx.x==0) {
		block_buffer = (int*)malloc(sizeof(unsigned int)*block_buffer_size);
	}
#elif BUFFER_ALLOCATOR == 1
	if (threadIdx.x==0) {
		block_buffer = (int*)hamalloc(sizeof(unsigned int)*block_buffer_size);
	}
#else
	if (threadIdx.x==0) {
		block_buffer = &buffer[block_offset];
	}
#endif 
    int t_idx = 0;                                          // used to access the buffer
    if (threadIdx.x == 0) *block_index = 0;
    __syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<node_num){
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				dot += data[i] * x[indices[i]];
			}
			y[tid] = dot;
		}
		else {
			t_idx = atomicInc(block_index, block_buffer_size);
			block_buffer[t_idx] = tid;
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
		spmv_process_buffer<<<block_num,NESTED_BLOCK_SIZE,0,s>>>( ptr, indices, data, x, y, node_num,
																	buffer, *block_index);
#if BUFFER_ALLOCATOR == 0 // default 
		cudaDeviceSynchronize();
		free(block_buffer);
#elif BUFFER_ALLOCATOR == 1
		cudaDeviceSynchronize();
		hafree(block_buffer);
#endif

	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void csr_spmv_grid_dp_kernel(int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x,
										FLOAT_T *y, int node_num, int *buffer,
										unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	int t_idx = 0;						// used to access the buffer
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<node_num){
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				dot += data[i] * x[indices[i]];
			}
			y[tid] = dot;
		}
		else {
			t_idx = atomicInc(idx, GM_BUFF_SIZE);
			buffer[t_idx] = tid;
		}
	}
	// 2nd phase, grid level consolidation
	if (threadIdx.x==0) {
		// count up
		if ( atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) ) {//
			//printf("gridDim.x: %d buffer: %d\n", gridDim.x, *idx);
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
#endif
			dim3 dimGridB(1,1,1);
			dimGridB.x = 208;
			if ( *idx<dimGridB.x ) dimGridB.x = *idx;
			csr_spmv_block_queue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(ptr, indices, buffer, idx,
																			data, x, y, node_num );
		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void csr_spmv_grid_dp_complex_kernel(int *ptr, int *indices, FLOAT_T *data, FLOAT_T *x,
										FLOAT_T *y, int node_num, int *buffer,
										unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
	__shared__ int shm_buffer[MAXDIMBLOCK];
	__shared__ unsigned int block_index;				// index of each block within its sub-buffer
	__shared__ int offset;
	int t_idx = 0;						// used to access the buffer
	if (threadIdx.x == 0) block_index = 0;
	__syncthreads();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<node_num){
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				dot += data[i] * x[indices[i]];
			}
			y[tid] = dot;
		}
		else {
			t_idx = atomicInc(&block_index, per_block_buffer);
			shm_buffer[t_idx] = tid;
		}
	}
	__syncthreads();
	// reorganize consolidation buffer for load balance (get offset per block)
	if (threadIdx.x==0) {
		offset = atomicAdd(idx, block_index);
	}
	__syncthreads();
	// dump shm_buffer to global buffer
	if (threadIdx.x<block_index) {
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
			csr_spmv_block_queue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(ptr, indices, buffer, idx,
																			data, x, y, node_num );
		}
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
