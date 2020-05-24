///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <memory>

#define IMG_HEIGHT 64
#define IMG_WIDTH 64

#define N_COLORS 4

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#ifndef DEBUG
#define dbg_printf(...)
#else
#define dbg_printf(...) do { printf(__VA_ARGS__); } while (0)
#endif

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

void cpu_process(uchar *img_in, uchar *img_out, int width, int height);

/* Abstract base class for both parts of the exercise */
class image_processing_server
{
public:
    virtual ~image_processing_server() {}

    /* Enqueue an image for processing. Receives pointers to pinned host
     * memory. Return false if there is no room for image (caller will try again).
     */
    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) = 0;

    /* Checks whether an image has completed processing. If so, set img_id
     * accordingly, and return true. */
    virtual bool dequeue(int *img_id) = 0;
};

std::unique_ptr<image_processing_server> create_streams_server();
std::unique_ptr<image_processing_server> create_queues_server(int threads);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

