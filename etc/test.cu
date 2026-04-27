#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 16384  // matrix size NxN, increase to push harder

__global__ void matmul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    
    // allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // fill with random data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    printf("Running %dx%d matrix multiplication...\n", N, N);
    
    clock_t start = clock();
    matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    clock_t end = clock();

    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
    
    printf("Time: %.2f ms\n", ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
