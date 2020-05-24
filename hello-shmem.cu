#include <iostream>
#include <cuda/atomic>

class shared_memory
{
private:
    // On each use of the flag, we switch the meaning of its values (true/false -> data is ready/not-ready)
    bool reader_next_flag;
    bool writer_next_flag;
    cuda::atomic<bool> flag;
    char data;

public:
    shared_memory() :
        reader_next_flag(true),
        writer_next_flag(true),
        flag(false),
        data(-1)
    {}

    __device__ __host__ char read_message()
    {
        while (flag.load(cuda::memory_order_acquire) != reader_next_flag)
            ;

        reader_next_flag = !reader_next_flag;

        return data;
    }

    __device__ __host__ void write_message(char value)
    {
        data = value;
        flag.store(writer_next_flag, cuda::memory_order_release);

        writer_next_flag = !writer_next_flag;
    }
};

__global__ void kernel(shared_memory* shmem_input, shared_memory* shmem_output)
{
    char c;
    do {
        c = shmem_input->read_message();
        shmem_output->write_message(c);
    } while (c);
}

int main(int argc, char* argv[]) {
    char *pinned_host_buffer;

    // Allocate pinned host buffer for two shared_memory instances
    cudaMallocHost(&pinned_host_buffer, 2 * sizeof(shared_memory));
    // Use placement new operator to construct our class on the pinned buffer
    shared_memory *shmem_host_to_gpu = new (pinned_host_buffer) shared_memory();
    shared_memory *shmem_gpu_to_host = new (pinned_host_buffer + sizeof(shared_memory)) shared_memory();

    bool verbose = true;
    std::string message_to_gpu = "Hello shared memory!";
    size_t msg_len = message_to_gpu.length();

    if (argc > 1) {
        msg_len = atoi(argv[1]);
        message_to_gpu.resize(msg_len);
        for (size_t i = 0; i < msg_len; ++i)
            message_to_gpu[i] = i & 0xff | 1;
        verbose = false;
    }

    auto message_from_gpu = std::string(msg_len, '\0');

    // Invoke kernel asynchronously
    kernel<<<1, 1>>>(shmem_host_to_gpu, shmem_gpu_to_host);

    std::cout << "Writing message to GPU:" << std::endl;

    for (size_t i = 0; i < msg_len; ++i) {
        char c = message_to_gpu[i];
        shmem_host_to_gpu->write_message(c);
        message_from_gpu[i] = shmem_gpu_to_host->read_message();
        if (verbose)
            std::cout << c << std::flush;
    }
    shmem_host_to_gpu->write_message(0);

    if (verbose)
        std::cout << "\nresult:\n" << message_from_gpu << std::endl;
    std::cout << "\n" << "Waiting for kernel to complete." << std::endl;

    cudaError_t err = cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(err) << std::endl;

    if (message_from_gpu != message_to_gpu) {
        std::cout << "Error: got different string from GPU." << std::endl;
    }

    // Destroy queues and release memory
    shmem_host_to_gpu->~shared_memory();
    shmem_gpu_to_host->~shared_memory();
    err = cudaFreeHost(pinned_host_buffer);
    assert(err == cudaSuccess);

    cudaDeviceReset();

    return 0;
}



