// CPU baseline: the same int8 20-16-2 MLP the FPGA runs, with identical integer math.
//
//   gcc -O2 host/bench_cpu.c -o build/bench_cpu_O2
//   build/bench_cpu_O2 model/weights
//
// Checks all 5 test examples against z2_reference.txt, then times ITERS single
// inferences one at a time with the CPU timestamp counter (rdtsc), so the output
// shows the spread (jitter) as well as the typical latency. Prints one JSON object.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <windows.h>

#define K1 20
#define H 16
#define O 2
#define M 5
#define ITERS 1000000

static int8_t W1[K1][H], b1[H], W2[H][O], b2[O], X[M][K1];
static int32_t z2_ref[M][O];

static void load(const char *dir, const char *name, int n, int is_int8, void *dst) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "can't open %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) {
        long v;
        if (fscanf(f, "%ld", &v) != 1) { fprintf(stderr, "short read in %s\n", path); exit(1); }
        if (is_int8) ((int8_t *)dst)[i] = (int8_t)v;
        else         ((int32_t *)dst)[i] = (int32_t)v;
    }
    fclose(f);
}

// z1 = x @ W1 + b1*64 -> ReLU -> requant (z + 32) >> 6, cap 127 -> z2 = a1 @ W2 + b2*64
static void infer(const int8_t x[K1], int32_t z2[O]) {
    int8_t a1[H];
    for (int j = 0; j < H; j++) {
        int32_t acc = b1[j] * 64;
        for (int k = 0; k < K1; k++)
            acc += x[k] * W1[k][j];
        if (acc < 0) acc = 0;
        acc = (acc + 32) >> 6;
        a1[j] = acc > 127 ? 127 : acc;
    }
    for (int o = 0; o < O; o++) {
        int32_t acc = b2[o] * 64;
        for (int j = 0; j < H; j++)
            acc += a1[j] * W2[j][o];
        z2[o] = acc;
    }
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "model/weights";
    load(dir, "W1.txt", K1 * H, 1, W1);
    load(dir, "b1.txt", H, 1, b1);
    load(dir, "W2.txt", H * O, 1, W2);
    load(dir, "b2.txt", O, 1, b2);
    load(dir, "X_test.txt", M * K1, 1, X);
    load(dir, "z2_reference.txt", M * O, 0, z2_ref);

    for (int m = 0; m < M; m++) {
        int32_t z2[O];
        infer(X[m], z2);
        if (z2[0] != z2_ref[m][0] || z2[1] != z2_ref[m][1]) {
            fprintf(stderr, "example %d wrong: got %d %d, expected %d %d\n",
                    m, z2[0], z2[1], z2_ref[m][0], z2_ref[m][1]);
            return 1;
        }
    }

    // timestamp-counter frequency, calibrated against the OS high-resolution timer
    LARGE_INTEGER qf, q0, q1;
    QueryPerformanceFrequency(&qf);
    QueryPerformanceCounter(&q0);
    uint64_t c0 = __rdtsc();
    Sleep(200);
    QueryPerformanceCounter(&q1);
    uint64_t c1 = __rdtsc();
    double tsc_hz = (double)(c1 - c0) * qf.QuadPart / (double)(q1.QuadPart - q0.QuadPart);

    uint64_t *ticks = malloc(sizeof(uint64_t) * ITERS);
    volatile int32_t sink = 0;
    unsigned aux;

    // cost of the timing itself, subtracted from every sample
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = __rdtscp(&aux);
        uint64_t t1 = __rdtscp(&aux);
        ticks[i] = t1 - t0;
    }
    qsort(ticks, ITERS, sizeof *ticks, cmp_u64);
    uint64_t overhead = ticks[ITERS / 2];

    for (int i = 0; i < 10000; i++) {   // warm up caches and branch predictors
        int32_t z2[O];
        infer(X[i % M], z2);
        sink ^= z2[0];
    }

    uint64_t start = __rdtsc();
    for (int i = 0; i < ITERS; i++) {
        int32_t z2[O];
        uint64_t t0 = __rdtscp(&aux);
        infer(X[i % M], z2);
        uint64_t t1 = __rdtscp(&aux);
        sink ^= z2[0] ^ z2[1];
        ticks[i] = (t1 - t0 > overhead) ? t1 - t0 - overhead : 0;
    }
    uint64_t total = __rdtsc() - start;
    qsort(ticks, ITERS, sizeof *ticks, cmp_u64);

    double ns = 1e9 / tsc_hz;
    #define PCT(p) (ticks[(size_t)((p) / 100.0 * (ITERS - 1))] * ns)
    printf("{\"iters\": %d, \"tsc_ghz\": %.4f, \"overhead_ns\": %.2f, "
           "\"min_ns\": %.2f, \"p50_ns\": %.2f, \"p90_ns\": %.2f, \"p99_ns\": %.2f, "
           "\"p999_ns\": %.2f, \"max_ns\": %.2f, \"loop_avg_ns\": %.2f, \"sink\": %d}\n",
           ITERS, tsc_hz / 1e9, overhead * ns,
           PCT(0), PCT(50), PCT(90), PCT(99), PCT(99.9), PCT(100),
           (double)total * ns / ITERS, sink);
    free(ticks);
    return 0;
}
