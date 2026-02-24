#include <stdio.h>
#include <stdlib.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
	for (int i = 0 ;i< n; ++i){
		C_h[i] = A_h[i] + B_h[i];
	}
}

int main(int argc, char* argv[]) {
	int n = atoi(argv[1]);

	float* A_h = (float*)malloc(n * sizeof(float));
	float* B_h = (float*)malloc(n * sizeof(float));
	float* C_h = (float*)malloc(n * sizeof(float));

	for (int i = 0; i < n; ++i)
		scanf("%f", &A_h[i]);

	for (int i = 0; i < n; ++i)
		scanf("%f", &B_h[i]);

	vecAdd(A_h, B_h, C_h, n);

	for (int i = 0; i < n; ++i)
		printf("%f ", C_h[i]);
	printf("\n");

	free(A_h);
	free(B_h);
	free(C_h);
}
