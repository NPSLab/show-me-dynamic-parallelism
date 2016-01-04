#ifndef __BFS_REC_KERNEL__
#define __BFS_REC_KERNEL__

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAX_THREAD_PER_BLOCK 1024

#define WARP_SIZE 32
#define SHM_BUFF_SIZE 256

#define GPU_PROFILE

#ifdef GPU_PROFILE
// records the number of kerbel calls performed
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
	printf("====> GPU #%u - number of kernel calls:%u\n",solution, nested_calls);
}

__global__ void reset_gpu_statistics(){
	nested_calls = 0;
}
#endif

//#if (CONSOLIDATE_LEVEL==2)

__device__ unsigned int tmp_buffer[GM_BUFF_SIZE];
__device__ unsigned int tmp_idx;

//#endif

__global__ void gpu_print(unsigned int *idx)
{
	printf("index: %d\n", *idx);

}

__device__ unsigned int gm_idx_pool[2000][1];

// iterative, flat BFS traversal (note: synchronization-free implementation)
__global__ void bfs_kernel_flat(int level, int num_nodes, int *vertexArray, int *edgeArray, int *levelArray, bool *queue_empty){
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) atomicInc(&nested_calls, INF);
#endif
	unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
	for (int node = tid; node < num_nodes; node +=blockDim.x * gridDim.x){
		if(node < num_nodes && levelArray[node]==level){
			for (int edge=vertexArray[node];edge<vertexArray[node+1];edge++){
				int neighbor=edgeArray[edge];
				if (levelArray[neighbor]==UNDEFINED || levelArray[neighbor]>(level+1)){
					levelArray[neighbor]=level+1;
					*queue_empty=false;
				}
			}	
		}
	}	
}

// recursive naive NFS traversal
__global__ void bfs_kernel_dp(int node, int *vertexArray, int *edgeArray, int *levelArray){
#ifdef GPU_PROFILE
	if (threadIdx.x+blockDim.x*blockIdx.x==0) atomicInc(&nested_calls, INF);
#endif

#if (STREAMS!=0)
	cudaStream_t s[STREAMS];
	for (int i=0; i<STREAMS; ++i)  cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);	
#endif

	int num_children = vertexArray[node+1]-vertexArray[node];
	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x){ // may change this to use multiple blocks
		int child = edgeArray[vertexArray[node]+childp];
		int node_level = levelArray[node];
		int child_level = levelArray[child];
		if (child_level==UNDEFINED || child_level>(node_level+1)){
			unsigned old_level = atomicMin(&levelArray[child],node_level+1);
			if (old_level == child_level){
				unsigned num_grandchildren=vertexArray[child+1]-vertexArray[child];
				unsigned block_size = min(num_grandchildren, THREADS_PER_BLOCK);
#if (STREAMS!=0)
			        if (block_size!=0) bfs_kernel_dp<<<1,block_size, 0, s[threadIdx.x%STREAMS]>>>(child, vertexArray, edgeArray, levelArray);
#else
			        if (block_size!=0) bfs_kernel_dp<<<1,block_size>>>(child, vertexArray, edgeArray, levelArray);
#endif
			}
		}
	}
}

// recursive hierarchical BFS traversal
__global__ void bfs_kernel_dp_hier(int node, int *vertexArray, int *edgeArray, int *levelArray){
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) atomicInc(&nested_calls, INF);
#endif

#if (STREAMS!=0)
	cudaStream_t s[STREAMS];
	for (int i=0; i<STREAMS; ++i)  cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);	
#endif
	__shared__ int child;
	__shared__ int child_level;
	__shared__ unsigned num_grandchildren;
	
	int node_level = levelArray[node];
	unsigned num_children = vertexArray[node+1]-vertexArray[node];
	
	for (unsigned childp = blockIdx.x; childp < num_children; childp+=gridDim.x){
		if (threadIdx.x==0){
			child = edgeArray[vertexArray[node]+childp];
			num_grandchildren = 0; // by default, do not continue
			child_level = levelArray[child];
			if (child_level==UNDEFINED || child_level>(node_level+1)){
				unsigned old_level = atomicMin(&levelArray[child],node_level+1);
				if (old_level == child_level)
					num_grandchildren = vertexArray[child+1]-vertexArray[child];
			}
		}
		__syncthreads();
		if (num_grandchildren != 0){
			for (unsigned grandchild_p = threadIdx.x; grandchild_p < num_grandchildren; grandchild_p+=blockDim.x){
				unsigned grandchild = edgeArray[vertexArray[child]+grandchild_p];
				unsigned grandchild_level = levelArray[grandchild];
				if (grandchild_level == UNDEFINED || grandchild_level > (node_level + 2)){
					unsigned old_level = atomicMin(&levelArray[grandchild],node_level+2);
					if (old_level == grandchild_level){
						unsigned num_grandgrandchildren = vertexArray[grandchild+1]-vertexArray[grandchild];
#if (STREAMS!=0)
						if (num_grandgrandchildren!=0) 
							bfs_kernel_dp_hier<<<num_grandgrandchildren,THREADS_PER_BLOCK, 0, s[threadIdx.x%STREAMS]>>>(grandchild, vertexArray, edgeArray, levelArray);
#else 
						if (num_grandgrandchildren!=0) 
							bfs_kernel_dp_hier<<<num_grandgrandchildren,THREADS_PER_BLOCK>>>(grandchild, vertexArray, edgeArray, levelArray);
#endif
					}
				}
			}
		}
		__syncthreads();
	}
}

// prepare bfs_kernel_dp_cons
__global__ void  bfs_kernel_dp_cons_prepare(int *levelArray, unsigned int *buffer, 
													unsigned *idx, int source)
{
	levelArray[source] = 0;		// redundant
	buffer[0] = source;
	*idx = 1;
	//printf("Source : %d\n", source);
	//printf("Buffer address : %p\n", buffer);
	//printf("LevelArray address : %p\n", levelArray);
	//printf("%d\n", (buffer-0x13001b3c28));
	//printf("%d\n", sizeof(unsigned int));
}

// recursive BFS traversal with warp-level consolidation
__global__ void bfs_kernel_dp_warp_cons(int *vertexArray, int *edgeArray, int *levelArray, 
										unsigned int *queue, unsigned int queue_size,
										unsigned int *buffer, unsigned int *idx) {
	unsigned int t_idx;
	__shared__ unsigned int sh_idx[THREADS_PER_BLOCK/WARP_SIZE+1];
	__shared__ unsigned int ori_idx[THREADS_PER_BLOCK/WARP_SIZE+1];
	__shared__ unsigned int total_num_child;

	int warp_id = threadIdx.x / WARP_SIZE;
//	int warp_dim = blockDim.x / WARP_SIZE;
//	int total_warp_num = gridDim.x * warp_dim;	
	if ( threadIdx.x==0 ) {
		total_num_child = 0;
		for ( unsigned bid = blockIdx.x; bid<queue_size; bid += gridDim.x) {
			int node = queue[bid];
			total_num_child += vertexArray[node+1]-vertexArray[node];
		}
	}
	__syncthreads();

	if (threadIdx.x%WARP_SIZE==0) {
		//ori_idx[warp_id] = atomicAdd(idx, num_children);
		ori_idx[warp_id] = atomicAdd(idx, total_num_child/(WARP_SIZE/16));
		sh_idx[warp_id] = ori_idx[warp_id];
	}
	
	for (unsigned bid = blockIdx.x; bid<queue_size; bid += gridDim.x ) {
		int node = queue[bid];
		unsigned int num_children = vertexArray[node+1]-vertexArray[node];
		for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
			int child = edgeArray[vertexArray[node]+childp];
			unsigned node_level = levelArray[node];
			unsigned child_level = levelArray[child];
			if (child_level==UNDEFINED || child_level>(node_level+1)){
				unsigned old_level = atomicMin(&levelArray[child], node_level+1);
				t_idx = atomicInc(&sh_idx[warp_id], GM_BUFF_SIZE);
				buffer[t_idx] = child;
			}
		}
	}

	if (threadIdx.x%WARP_SIZE==0 && sh_idx[warp_id]>ori_idx[warp_id]) {
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		unsigned int size = sh_idx[warp_id]-ori_idx[warp_id];
	//	printf("Launch kernel with %d - %d = %d blocks\n", sh_idx, ori_idx, sh_idx-ori_idx);
		unsigned int block_num = 13;
		if (size<block_num) block_num = size;
		bfs_kernel_dp_warp_cons<<<block_num, THREADS_PER_BLOCK>>>(vertexArray, 
									 	edgeArray, levelArray, buffer+ori_idx[warp_id], size,
										buffer, idx);
	}

	// no post work require
}

// recursive BFS traversal with warp-level consolidation
__global__ void bfs_kernel_dp_warp_cons_unlimited(int *vertexArray, int *edgeArray, int *levelArray, 
								unsigned int *queue, unsigned int *buffer, unsigned int *idx) {
	unsigned int bid = blockIdx.x; // 1-Dimensional grid configuration
	unsigned int t_idx;
	__shared__ unsigned int sh_idx[THREADS_PER_BLOCK/WARP_SIZE+1];
	__shared__ unsigned int ori_idx[THREADS_PER_BLOCK/WARP_SIZE+1];

	int warp_id = threadIdx.x / WARP_SIZE;
//	int warp_dim = blockDim.x / WARP_SIZE;
//	int total_warp_num = gridDim.x * warp_dim;	

	int node = queue[bid];

	unsigned int num_children = vertexArray[node+1]-vertexArray[node];
	if (threadIdx.x%WARP_SIZE==0) {
		ori_idx[warp_id] = atomicAdd(idx, num_children);
		sh_idx[warp_id] = ori_idx[warp_id];
	}

	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
		int child = edgeArray[vertexArray[node]+childp];
		unsigned node_level = levelArray[node];
		unsigned child_level = levelArray[child];
		if (child_level==UNDEFINED || child_level>(node_level+1)){
			unsigned old_level = atomicMin(&levelArray[child], node_level+1);
			t_idx = atomicInc(&sh_idx[warp_id], GM_BUFF_SIZE);
			buffer[t_idx] = child;
		}
	}

	if (threadIdx.x%WARP_SIZE==0 && sh_idx[warp_id]>ori_idx[warp_id]) {
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
	//	printf("Launch kernel with %d - %d = %d blocks\n", sh_idx, ori_idx, sh_idx-ori_idx);
		bfs_kernel_dp_warp_cons_unlimited<<<sh_idx[warp_id]-ori_idx[warp_id], THREADS_PER_BLOCK>>>(vertexArray, 
									 	edgeArray, levelArray, buffer+ori_idx[warp_id], 
										buffer, idx);
	}

	// no post work require
}

// recursive BFS traversal with block-level consolidation
__global__ void bfs_kernel_dp_warp_malloc_cons(int *vertexArray, int *edgeArray, int *levelArray, 
								unsigned int *queue, unsigned int *buffer, unsigned int *idx) {
	unsigned int bid = blockIdx.x; // 1-Dimensional grid configuration
	unsigned int t_idx;
	__shared__ unsigned int sh_idx[THREADS_PER_BLOCK/WARP_SIZE+1];
	__shared__ unsigned int* sh_buffer[THREADS_PER_BLOCK/WARP_SIZE+1];

	int warp_id = threadIdx.x / WARP_SIZE;
//	int warp_dim = blockDim.x / WARP_SIZE;
//	int total_warp_num = gridDim.x * warp_dim;	

	int node = queue[bid];

	unsigned int num_children = vertexArray[node+1]-vertexArray[node];
	if (threadIdx.x%WARP_SIZE==0) {
		sh_buffer[warp_id] = (unsigned int*)malloc(sizeof(unsigned int)*num_children);
		sh_idx[warp_id] = 0;
	}

	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
		int child = edgeArray[vertexArray[node]+childp];
		unsigned node_level = levelArray[node];
		unsigned child_level = levelArray[child];
		if (child_level==UNDEFINED || child_level>(node_level+1)){
			unsigned old_level = atomicMin(&levelArray[child], node_level+1);
			t_idx = atomicInc(&sh_idx[warp_id], GM_BUFF_SIZE);
			buffer[t_idx] = child;
		}
	}

	if (threadIdx.x%WARP_SIZE==0 && sh_idx[warp_id]>0) {
	//	printf("Launch kernel with %d - %d = %d blocks\n", sh_idx, ori_idx, sh_idx-ori_idx);
		bfs_kernel_dp_warp_malloc_cons<<<sh_idx[warp_id], THREADS_PER_BLOCK>>>(vertexArray, 
									 	edgeArray, levelArray, sh_buffer[warp_id], 
										buffer, idx);
#ifdef FORCE_SYNC
		cudaDeviceSynhronize();
		free(sh_buffer[warp_id]);
#endif	
	}

	// no post work require
}

// recursive BFS traversal with block-level consolidation
__global__ void bfs_kernel_dp_block_cons(int *vertexArray, int *edgeArray, int *levelArray, 
										unsigned int *queue, unsigned int queue_size, 
										unsigned int *buffer, unsigned int *idx) {
	unsigned int t_idx;
	__shared__ unsigned int sh_idx;
	__shared__ unsigned int ori_idx;
	__shared__ unsigned int total_num_child;
	if (threadIdx.x==0) {
		total_num_child = 0;
		for (unsigned bid = blockIdx.x; bid < queue_size; bid += gridDim.x) {
			int node = queue[bid];
			total_num_child += vertexArray[node+1]-vertexArray[node];
		}
		ori_idx = atomicAdd(idx, total_num_child);
		sh_idx = ori_idx;
	}
	__syncthreads();

	for (unsigned bid = blockIdx.x; bid < queue_size; bid += gridDim.x) {
		int node = queue[bid];
		unsigned num_children = vertexArray[node+1]-vertexArray[node];
		for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
			int child = edgeArray[vertexArray[node]+childp];
			unsigned child_level = levelArray[child];
			unsigned node_level = levelArray[node];
			if (child_level==UNDEFINED || child_level>(node_level+1)){
				unsigned old_level = atomicMin(&levelArray[child], node_level+1);
				t_idx = atomicInc(&sh_idx, GM_BUFF_SIZE);
				buffer[t_idx] = child;
			}
		}
	}
	__syncthreads();
	if (threadIdx.x==0 && sh_idx>ori_idx) {
#ifdef GPU_PROFILE
		atomicInc(&nested_calls, INF);
#endif
		//printf("Launch kernel with %d - %d = %d blocks\n", sh_idx, ori_idx, sh_idx-ori_idx);
		//bfs_kernel_dp_block_cons<<<sh_idx-ori_idx, THREADS_PER_BLOCK>>>(vertexArray, 
		unsigned int block_num = 13;
		if (sh_idx-ori_idx<block_num) block_num = sh_idx-ori_idx;
		bfs_kernel_dp_block_cons<<<13, THREADS_PER_BLOCK>>>(vertexArray, 
									 	edgeArray, levelArray, buffer+ori_idx, sh_idx-ori_idx, 
										buffer, idx);
	}
	
	// no post work require
}

// recursive BFS traversal with block-level consolidation
__global__ void bfs_kernel_dp_block_malloc_cons(int *vertexArray, int *edgeArray, int *levelArray, 
								unsigned int *queue, unsigned int *buffer, unsigned int *idx) {
	unsigned int bid = blockIdx.x; // 1-Dimensional grid configuration
	unsigned int t_idx;
	__shared__ unsigned int sh_idx;
	__shared__ unsigned int *sh_buffer;
	int node = queue[bid];

	unsigned int num_children = vertexArray[node+1]-vertexArray[node];
	if (threadIdx.x==0) {
		sh_buffer = (unsigned int*)malloc(sizeof(unsigned int)*num_children);
		sh_idx = 0;
	}
	__syncthreads();

	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
		int child = edgeArray[vertexArray[node]+childp];
		unsigned node_level = levelArray[node];
		unsigned child_level = levelArray[child];
		if (child_level==UNDEFINED || child_level>(node_level+1)){
			unsigned old_level = atomicMin(&levelArray[child], node_level+1);
			t_idx = atomicInc(&sh_idx, GM_BUFF_SIZE);
			sh_buffer[t_idx] = child;
		}
	}
	__syncthreads();
	if (threadIdx.x==0 && sh_idx>0) {
	//	printf("Launch kernel with %d - %d = %d blocks\n", sh_idx, ori_idx, sh_idx-ori_idx);
		bfs_kernel_dp_block_malloc_cons<<<sh_idx, THREADS_PER_BLOCK>>>(vertexArray, 
									 	edgeArray, levelArray, sh_buffer, 
										buffer, idx);
#ifdef FORCE_SYNC
		cudaDeviceSynchronize();
		free(sh_buffer);
#endif
	}
	// no post work require
}

__global__ void dp_grid_cons_init()
{
	tmp_idx = 0;
}

// recursive BFS traversal with grid-level consolidation
// queue and buffer work like Ping-Pong pointer
__global__ void bfs_kernel_dp_grid_cons(int *vertexArray, int *edgeArray, int *levelArray, 
									unsigned int *queue, unsigned int *qidx, 
									unsigned int *buffer, unsigned int *idx,
									unsigned int *count) 
{
	unsigned int bid = blockIdx.x; //+ blockIdx.y*gridDim.x; // 1-Dimensional grid configuration
	unsigned int t_idx;
	__shared__ unsigned int *sh_buffer;
	__shared__ unsigned int sh_idx;
	__shared__ unsigned int ori_idx;
	__shared__ unsigned int offset;
	for ( ; bid<*qidx; bid += gridDim.x ) {
		int node = queue[bid];

		unsigned int num_children = vertexArray[node+1]-vertexArray[node];

		for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
			int child = edgeArray[vertexArray[node]+childp];
			unsigned node_level = levelArray[node];
			unsigned child_level = levelArray[child];
			if (child_level==UNDEFINED || child_level>(node_level+1)){
				unsigned old_level = atomicMin(&levelArray[child], node_level+1);
				t_idx = atomicInc(idx, GM_BUFF_SIZE);
				//sh_buffer[t_idx] = child;
				buffer[t_idx] = child;
			}
		}
	}
	__syncthreads();

	// 2nd phase, grid level kernel launch
	if (threadIdx.x==0) {
		// count up
		if (atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) ) {
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
#endif
			//printf("Buffer size %d\n", *idx);
			*count = 0;	// reset counter
			*qidx = 0;	// reset next buffer index
			tmp_idx = 0;
			dim3 dimGrid(1,1,1);
			dimGrid.x = 13 * 16;
			if (*idx<=208) 	dimGrid.x = *idx;

			bfs_kernel_dp_grid_cons<<<dimGrid, THREADS_PER_BLOCK>>>(vertexArray, edgeArray,
								levelArray, buffer, idx, queue, qidx, count);
	
//			bfs_kernel_dp_grid_cons<<<dimGrid, THREADS_PER_BLOCK>>>(vertexArray, edgeArray, 
//								levelArray, buffer+ori_idx, qidx, buffer, idx, count);
#ifdef FORCE_SYNC
			cudaDeviceSynchronize();
#endif
		}
	}
	// no post work require
}

// recursive BFS traversal with grid-level consolidation
__global__ void bfs_kernel_dp_grid_malloc_cons(int *vertexArray, int *edgeArray, int *levelArray, 
								unsigned int *queue, unsigned int *qidx, 
								unsigned int *buffer, unsigned int *idx,
								unsigned int *count) 
{
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nestd_calls++;
#endif
	unsigned int bid = blockIdx.x; // 1-Dimensional grid configuration
	unsigned int t_idx;
	__shared__ int *sh_buffer;
	__shared__ unsigned int sh_idx;
	//__shared__ unsigned int ori_idx;
	__shared__ unsigned int offset;
	int node = queue[bid];

	unsigned int num_children = vertexArray[node+1]-vertexArray[node];
	if (threadIdx.x==0) {
		sh_buffer = (int*)malloc(sizeof(int)*num_children);
		sh_idx = 0;
		//ori_idx = atomicAdd(idx, num_children);
		//sh_idx = ori_idx;
	}
	__syncthreads();

	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
		int child = edgeArray[vertexArray[node]+childp];
		unsigned node_level = levelArray[node];
		unsigned child_level = levelArray[child];
		if (child_level==UNDEFINED || child_level>(node_level+1)){
			unsigned old_level = atomicMin(&levelArray[child], node_level+1);
			t_idx = atomicInc(&sh_idx, GM_BUFF_SIZE);
			sh_buffer[t_idx] = child;
		}
	}
	__syncthreads();
	// reorganize consolidation buffer for load balance ()
	if (threadIdx.x==0) {
		//offset = atomicAdd(qidx, sh_idx-ori_idx);
		offset = atomicAdd(idx, sh_idx);
	}
	__syncthreads();
	// dump block_level buffer to grids
	for (unsigned tid = threadIdx.x; tid<sh_idx; tid+=blockDim.x) {
		int gm_idx = tid + offset;
		buffer[gm_idx] = sh_buffer[tid];
	}
	__syncthreads();

	// 2nd phase, grid level kernel launch
	if (threadIdx.x==0) {
		free(sh_buffer);	// free allocated block buffer
		// count up
		if (atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) && *idx!=0 ) {
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
#endif
			printf("Buffer size %d\n", *idx);
//			*count = malloc(sizeof(unsigned int));
//			*qidx = malloc(sizeof(unsigned int));
			*count = 0;	// reset counter
			*qidx = 0;	// reset next buffer index
			dim3 dimGrid(1,1,1);
			if (*idx<=MAXDIMGRID) {
				dimGrid.x = *idx;
			}
		/*	else if (*idx<=MAXDIMGRID*THREADS_PER_BLOCK) {
				dimGrid.x = MAXDIMGRID;
				dimGrid.y = *idx/MAXDIMGRID+1;
			}*/
			else {
				printf("Too many elements in queue\n");
			}

			bfs_kernel_dp_grid_malloc_cons<<<dimGrid, THREADS_PER_BLOCK>>>(vertexArray, edgeArray,
								levelArray, buffer, idx, queue, qidx, count);
	
#ifdef FORCE_SYNC
			cudaDeviceSynchronize();
			free(sh_buffer);
#endif
		}
	}

//	if (threadIdx.x==0 && sh_idx>ori_idx) {
	//	printf("Launch kernel with %d - %d = %d blocks\n", sh_idx, ori_idx, sh_idx-ori_idx);
//		bfs_kernel_dp_grid_cons<<<sh_idx-ori_idx, THREADS_PER_BLOCK>>>(vertexArray, 
//									 	edgeArray, levelArray, buffer+ori_idx, 
//										buffer, idx);
//	}

	// no post work require
}

// recursive BFS traversal with grid-level consolidation
// queue and buffer work like Ping-Pong pointer
__global__ void bfs_kernel_dp_grid_cons_complex(int *vertexArray, int *edgeArray, int *levelArray, 
									unsigned int *queue, unsigned int *qidx, 
									unsigned int *buffer, unsigned int *idx,
									unsigned int *count) 
{
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nestd_calls++;
#endif
	unsigned int bid = blockIdx.x; //+ blockIdx.y*gridDim.x; // 1-Dimensional grid configuration
	unsigned int t_idx;
	__shared__ unsigned int *sh_buffer;
	__shared__ unsigned int sh_idx;
	__shared__ unsigned int ori_idx;
	__shared__ unsigned int offset;
	int node = queue[bid];

	unsigned int num_children = vertexArray[node+1]-vertexArray[node];
	if (threadIdx.x==0) {
		ori_idx = atomicAdd(&tmp_idx, num_children);
		sh_idx = 0;
		sh_buffer = tmp_buffer+ori_idx;
	}
	__syncthreads();

	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x) {
		int child = edgeArray[vertexArray[node]+childp];
		unsigned node_level = levelArray[node];
		unsigned child_level = levelArray[child];
		if (child_level==UNDEFINED || child_level>(node_level+1)){
			unsigned old_level = atomicMin(&levelArray[child], node_level+1);
			t_idx = atomicInc(&sh_idx, GM_BUFF_SIZE);
			sh_buffer[t_idx] = child;
		}
	}
	__syncthreads();
	// reorganize consolidation buffer for load balance ()
	if (threadIdx.x==0) {
		//offset = atomicAdd(qidx, sh_idx-ori_idx);
		offset = atomicAdd(idx, sh_idx);
	}
	__syncthreads();
	// dump block-level buffer to grid-level buffer
	for (unsigned tid = threadIdx.x; tid<sh_idx; tid+=blockDim.x) {
		int gm_idx = tid + offset;
		buffer[gm_idx] = sh_buffer[tid];
	}
	__syncthreads();

	// 2nd phase, grid level kernel launch
	if (threadIdx.x==0) {
		// count up
		if (atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) ) {
#ifdef GPU_PROFILE
			atomicInc(&nested_calls, INF);
#endif
			//printf("Buffer size %d\n", *idx);
			*count = 0;	// reset counter
			*qidx = 0;	// reset next buffer index
			tmp_idx = 0;
			dim3 dimGrid(1,1,1);
			if (*idx<=MAXDIMGRID) {
				dimGrid.x = *idx;
			}
			/*else if (*idx<=MAXDIMGRID*THREADS_PER_BLOCK) {
				dimGrid.x = MAXDIMGRID;
				dimGrid.y = *idx/MAXDIMGRID+1;
			}*/
			else {
				printf("%d \n", *idx);
				printf("Too many elements in queue\n");
			}

			bfs_kernel_dp_grid_cons<<<dimGrid, THREADS_PER_BLOCK>>>(vertexArray, edgeArray,
								levelArray, buffer, idx, queue, qidx, count);
	
//			bfs_kernel_dp_grid_cons<<<dimGrid, THREADS_PER_BLOCK>>>(vertexArray, edgeArray, 
//								levelArray, buffer+ori_idx, qidx, buffer, idx, count);
#ifdef FORCE_SYNC
			cudaDeviceSynchronize();
#endif
		}
	}
	// no post work require
}


#endif
