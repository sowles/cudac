#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		C[i] = A[i] + B[i];
}

static void checkCuda(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK(x) checkCuda((x), __FILE__, __LINE__)

int main(int argc, char* argv[]) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <n>\n", argv[0]);
		return EXIT_FAILURE;
	}

	int n = atoi(argv[1]);
	if (n <= 0) {
		fprintf(stderr, "n must be positive\n");
		return EXIT_FAILURE;
	}

	float* A_h = (float*)malloc(n * sizeof(float));
	float* B_h = (float*)malloc(n * sizeof(float));
	float* C_h = (float*)malloc(n * sizeof(float));

	for (int i = 0; i < n; ++i) {
		printf("A dimension %d: ", i + 1);
		scanf("%f", &A_h[i]);
	}

	for (int i = 0; i < n; ++i) {
		printf("B dimension %d: ", i + 1);
		scanf("%f", &B_h[i]);
	}

	float *d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_C, n * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_A, A_h, n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B_h, n * sizeof(float), cudaMemcpyHostToDevice));

	const int threadsPerBlock = 256;
	const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(C_h, d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

	printf("Result: ");
	for (int i = 0; i < n; ++i)
		printf("%f ", C_h[i]);
	printf("\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A_h);
	free(B_h);
	free(C_h);

	return 0;
}
