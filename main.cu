///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex2.h"

#include <random>
#include <algorithm>
#include <cassert>

#define SQR(a) ((a) * (a))
#define N_IMAGES 10000

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        if (img_arr1[i] != img_arr2[i])
            dbg_printf("cpu[0x%4x/0x%04x] == 0x%x != 0x%x\n", i / (IMG_WIDTH * IMG_HEIGHT), i % (IMG_WIDTH * IMG_HEIGHT), img_arr1[i], img_arr2[i]);
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

/* we'll use these to rate limit the request load */
struct rate_limit {
    double last_checked;
    double lambda; /* Requests per microsecond */
    std::default_random_engine generator;
    std::poisson_distribution<> distribution;

    /* Load in requests per second */
    rate_limit(double load, int seed) :
        last_checked(-1),
        lambda(load / 1e6),
        generator(seed),
        distribution(lambda)
    {
    }

    double number_to_send()
    {
        if (lambda == 0)
            return 1;

        if (last_checked < 0) {
            last_checked = get_time_msec() * 1e3;
            return 0;
        }

        double now = get_time_msec() * 1e3;
        double dt = now - last_checked;
        last_checked = now;
        int k = distribution(generator);
        return k * dt;
    }
};

int randomize_images(uchar *images)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0,0xffffffffffffffffULL);
    for (uint64_t *p = (uint64_t *)images; p < (uint64_t *)(images + N_IMAGES * IMG_WIDTH * IMG_HEIGHT); ++p)
	*p = distribution(generator);
    return 0;
} 

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}

enum program_mode {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};

program_mode parse_arguments(int argc, char **argv, int *threads, double *load)
{
    program_mode mode;
    
    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        *load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        *threads = atoi(argv[2]);
        *load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    return mode;
}

int main(int argc, char **argv) {
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    program_mode mode = parse_arguments(argc, argv, &threads_queue_mode, &load);

    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu; //output of GPU computation. In CPU memory.
    int devices;
    CUDA_CHECK( cudaGetDeviceCount(&devices) );
    printf("Number of devices: %d\n", devices);

    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    double t_start, t_finish;

    /* instead of loading real images, we'll load the arrays with random data */
    printf("\n=== Randomizing images ===\n");
    t_start = get_time_msec();
    if (randomize_images(images_in))
	return 1;
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        cpu_process(img_in, img_out, IMG_WIDTH, IMG_HEIGHT);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    printf("\n=== Client-Server ===\n");
    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %.1lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);

    long long int distance_sqr;
    std::unique_ptr<image_processing_server> server;

    switch (mode) {
    case PROGRAM_MODE_STREAMS:
        server = create_streams_server();
        break;
    case PROGRAM_MODE_QUEUE:
        server = create_queues_server(threads_queue_mode);
        break;
    }

    std::vector<double> req_t_start(N_IMAGES, NAN), req_t_end(N_IMAGES, NAN);

    rate_limit rate_limiter(load, 0);

    memset(images_out_gpu, 0, N_IMAGES * IMG_WIDTH * IMG_HEIGHT);

    t_start = get_time_msec();
    int next_img_id = 0;
    int num_dequeued = 0;
    double available_tasks = 0;

    while (next_img_id < N_IMAGES || num_dequeued < N_IMAGES) {
        int dequeued_img_id;
        if (server->dequeue(&dequeued_img_id)) {
            ++num_dequeued;
            req_t_end[dequeued_img_id] = get_time_msec();
        }

        /* If we are done with enqueuing, just loop until all are dequeued */
        if (next_img_id == N_IMAGES)
            continue;

        /* Toss a coin to see if we can enqueue more tasks under the currently simulated load */
        if (available_tasks < N_IMAGES) {
            double num_to_send = rate_limiter.number_to_send();
            for (int img_id = ceil(available_tasks); img_id < min(double(N_IMAGES), available_tasks + num_to_send); ++img_id) {
                req_t_start[img_id] = get_time_msec();
	    }
            available_tasks += num_to_send;
        }

        if (available_tasks > next_img_id) {
            /* Enqueue a new image */
            if (server->enqueue(next_img_id, &images_in[next_img_id * IMG_WIDTH * IMG_HEIGHT],
                                             &images_out_gpu[next_img_id * IMG_WIDTH * IMG_HEIGHT])) {
                ++next_img_id;
            }
        }
    }
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu);
    printf("distance from baseline %lld (should be zero)\n", distance_sqr);
    printf("throughput = %.1lf (req/sec)\n", N_IMAGES / (t_finish - t_start) * 1e+3);

    double latencies[N_IMAGES];
    double avg_latency = 0;
    for (int i = 0; i < N_IMAGES; i++) {
        double cur_latency = req_t_end[i] - req_t_start[i];
	assert(!isnan(req_t_start[i]));
	assert(!isnan(req_t_end[i]));
	latencies[i] = cur_latency;
        avg_latency += cur_latency;
    }
    avg_latency /= N_IMAGES;
    std::sort(latencies, latencies + N_IMAGES);

    printf("latency [msec]:\n%12s%12s%12s%12s%12s\n", "avg", "min", "median", "99th perc.", "max");
    printf("%12.4lf%12.4lf%12.4lf%12.4lf%12.4lf\n", avg_latency, latencies[0], latencies[N_IMAGES / 2], latencies[N_IMAGES * 99 / 100], latencies[N_IMAGES - 1]);

    return 0;
}
