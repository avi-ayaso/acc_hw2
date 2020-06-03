#include <iostream>
#include <cuda/atomic>
#include "ex2.h"
#define LOG_N_SLOTS 4
#define N_STREAMS 64
#define N_REGS 32
#define STREAM_AVAILABLE -1
#define END_RUN -2
#define SHMEM_PER_BLOCK 2577
#define REGS_PER_BLOCK 28

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

// Example single-threadblock kernel for processing a single image.
// Feel free to change it.
__global__ void process_image_kernel(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ uchar map[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < IMG_HEIGHT * IMG_HEIGHT; i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        float map_value = float(histogram[tid]) / (IMG_WIDTH * IMG_HEIGHT);
        map[tid] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }

    __syncthreads();

    for (int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += blockDim.x) {
        out[i] = map[in[i]];
    }
}




class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
	cudaStream_t streams[N_STREAMS];

    // Feel free to change the existing memory buffer definitions.
    int img_id_in_streams[N_STREAMS];
	uchar *dimg_in;
    uchar *dimg_out;

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
		
		for (int stream_idx = 0; stream_idx < N_STREAMS; stream_idx++) {
			img_id_in_streams[stream_idx] = STREAM_AVAILABLE;
			CUDA_CHECK(cudaStreamCreate(&streams[stream_idx]));
		}
		
        CUDA_CHECK( cudaMalloc(&dimg_in, N_STREAMS * IMG_WIDTH * IMG_HEIGHT) );
        CUDA_CHECK( cudaMalloc(&dimg_out, N_STREAMS * IMG_WIDTH * IMG_HEIGHT) );
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
		
		for (int stream_idx = 0; stream_idx < N_STREAMS; stream_idx++) {
			CUDA_CHECK(cudaStreamDestroy(streams[stream_idx]));
		}
		
		CUDA_CHECK( cudaFree(dimg_in) );
        CUDA_CHECK( cudaFree(dimg_out) );
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
		
		for (int stream_idx = 0; stream_idx < N_STREAMS; stream_idx++) {
			if (img_id_in_streams[stream_idx] != STREAM_AVAILABLE) {
				continue;
			}
			img_id_in_streams[stream_idx] = img_id;
			int offset = stream_idx * IMG_WIDTH * IMG_HEIGHT;
			CUDA_CHECK( cudaMemcpyAsync(dimg_in + offset, img_in, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyHostToDevice, streams[stream_idx]));
			process_image_kernel<<<1, 1024, 0, streams[stream_idx]>>>(dimg_in + offset, dimg_out + offset);
			CUDA_CHECK( cudaMemcpyAsync(img_out, dimg_out + offset, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToHost, streams[stream_idx]));
//			printf("img_id: %d was enqueued to stream idx: %d\n",img_id,stream_idx);
			return true;
		}
		return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) streams for any completed requests.

		for (int stream_idx = 0; stream_idx < N_STREAMS; stream_idx++) {
			cudaError_t status = cudaStreamQuery(streams[stream_idx]);
			switch (status) {
			case cudaSuccess:
				*img_id = img_id_in_streams[stream_idx]; // TODO return the img_id of the request that was completed.
				if(*img_id == STREAM_AVAILABLE) {
					continue;
				}
				img_id_in_streams[stream_idx] = STREAM_AVAILABLE;
//				printf("img_id: %d finished and was dequeued from stream idx: %d\n",*img_id,stream_idx);
				return true;
			case cudaErrorNotReady:
				continue;
			default:
				CUDA_CHECK(status);
			}
		}
		return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

typedef struct request{
    int img_id ;
    uchar* img_in;
    uchar* img_out;
} request_context;



template <size_t size>
class ring_buffer{
private:
    static const size_t N = 1 << size;
    request_context _mailbox[N];
    cuda::atomic<size_t> _head;
    cuda::atomic<size_t> _tail;
    request_context failure;
public:
    ring_buffer(){
        _head = 0;
        _tail = 0;
        failure.img_id = -1;
        for(size_t i = 0 ; i < N ; i++){
            _mailbox[i].img_id = -1;
        }
    }
    __device__ __host__ bool push(request_context data){
        size_t tail = _tail.load(cuda::memory_order_relaxed);
        if(tail - _head.load(cuda::memory_order_acquire) == N) return false;
        _mailbox[_tail % N] = data;
        _tail.store(tail + 1 , cuda::memory_order_release);
        return true;
    }
    __device__ __host__ request_context pop(){
        request_context item;
        size_t head = _head.load(cuda::memory_order_relaxed);
        if(_tail.load(cuda::memory_order_acquire) == head) return failure;
        item = _mailbox[_head % N];
        _head.store(head + 1 , cuda::memory_order_release);
        return item;
    }
};


__device__ void process_image(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ uchar map[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < IMG_HEIGHT * IMG_HEIGHT; i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        float map_value = float(histogram[tid]) / (IMG_WIDTH * IMG_HEIGHT);
        map[tid] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }

    __syncthreads();

    for (int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += blockDim.x) {
        out[i] = map[in[i]];
    }
    __syncthreads();
}


__global__ void producer_consumer_kernel(ring_buffer<LOG_N_SLOTS>* cpu_to_gpu, ring_buffer<LOG_N_SLOTS>* gpu_to_cpu) {
    request_context req;
    __shared__ uchar* img_in;
    __shared__ uchar* img_out;
    __shared__ bool con;
    do{
        if(threadIdx.x == 0){
            req = cpu_to_gpu[blockIdx.x].pop();
            con = false;
            img_in = req.img_in;
            img_out = req.img_out;
            if(req.img_id == -1) con = true;
            else if(req.img_id == END_RUN) {
                return;
            }                
        }
        __syncthreads();
        if(con) continue;   
        process_image(img_in, img_out);
        if(threadIdx.x == 0){
            while(!gpu_to_cpu[blockIdx.x].push(req));
        }
        __syncthreads();
    }while(1);
}



int getNumOfBlocks(int threads) {
    cudaDeviceProp prop;
    int min_limit;
    int n_used_block_regs = threads * REGS_PER_BLOCK;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int shmem_block_limit = prop.sharedMemPerMultiprocessor / SHMEM_PER_BLOCK;
    int regs_block_limit = prop.regsPerMultiprocessor / n_used_block_regs;
    int thread_block_limit = prop.maxThreadsPerMultiProcessor / threads;
    min_limit = shmem_block_limit < regs_block_limit ? shmem_block_limit : regs_block_limit;
    min_limit = min_limit < thread_block_limit ? min_limit :thread_block_limit;
    return min_limit * prop.multiProcessorCount;
}

class queue_server : public image_processing_server
{
private:
    ring_buffer<LOG_N_SLOTS> *cpu_to_gpu;
    ring_buffer<LOG_N_SLOTS> *gpu_to_cpu;
    char* pinned_host_buffer;
    int n_thread_blocks;
public:
    queue_server(int threads)
    {
        
        // TODO initialize host state
        // TODO launch GPU producer-consumer kernel with given number of threads
        n_thread_blocks = getNumOfBlocks(threads);
        // Allocate pinned host buffer for two shared_memory instances
        CUDA_CHECK(cudaMallocHost(&pinned_host_buffer, 2 * n_thread_blocks * sizeof(ring_buffer<LOG_N_SLOTS>)));
        // Use placement new operator to construct our class on the pinned buffer
        cpu_to_gpu = new (pinned_host_buffer) ring_buffer<LOG_N_SLOTS>[n_thread_blocks];
        gpu_to_cpu = new (pinned_host_buffer + n_thread_blocks * sizeof(ring_buffer<LOG_N_SLOTS>)) ring_buffer<LOG_N_SLOTS>[n_thread_blocks]; 
        producer_consumer_kernel<<<n_thread_blocks , threads>>>(cpu_to_gpu, gpu_to_cpu);
        std::cout << "num_of_blocks" << n_thread_blocks << std::endl;
    }

    ~queue_server() override
    {
        request_context end_context;
        end_context.img_id = END_RUN;
        end_context.img_in = NULL;
        end_context.img_out = NULL;
        while(!cpu_to_gpu[0].push(end_context));
        CUDA_CHECK( cudaDeviceSynchronize() );
        CUDA_CHECK( cudaFreeHost(pinned_host_buffer) );

    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        request_context req;
        req.img_id = img_id;
        req.img_in = img_in;
        req.img_out = img_out;
        for(int i = 0 ; i < n_thread_blocks ; i++){
            if(cpu_to_gpu[i].push(req)){
                return true;
            } 
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        *img_id = 0; // TODO return the img_id of the request that was completed.
        for(int i = 0 ; i < n_thread_blocks ; i++){         
            request_context req = gpu_to_cpu[i].pop();
            *img_id = req.img_id;
            if(req.img_id >= 0) {
                return true; 
            }
        }
        return false;
        
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
