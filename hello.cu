#include <stdio.h>

__global__ void addKernel(int *result){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	result[i] = i*2;
}

int main() {
	int result[0];
	int *d_result;
	cudaMalloc(&d_result, 8 * sizeof(int));
	addKernel<<<2, 4>>>(d_result);
	cudaMemcpy(result, d_result, 8 * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 8; i++)
		printf("result[%d] = %d\n", i, result[i]);
	cudaFree(d_result);
	return 0;
}
