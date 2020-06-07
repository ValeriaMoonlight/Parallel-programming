#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>


void Error(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void kernel(int *a, int *b, int *c, int count) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	c[i * count + j] = 0;
	for (int k = 0; k < count; k++) {
		c[i * count + j] += a[i * count + k] * b[k * count + j];
	}
}

int main() {

	srand(time(0));
	int N = 4;
	int size = 2;
	dim3 block(N / size, N / size);
	dim3 threads(size, size);
	int *dev_c, *dev_a, *dev_b;
	int *a, *b, *c;
	a = new int[N * N];
	b = new int[N * N];
	c = new int[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i*N + j] = rand() % 10;
			b[i*N + j] = rand() % 10;
		}
	}

	//std::cout << "A" << std::endl;
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		std::cout << a[i*N + j] << "  ";
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << "B" << std::endl;
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		std::cout << b[i*N + j] << "  ";
	//	}
	//	std::cout << std::endl;
	//}

	int time = GetTickCount();
	Error(cudaMalloc((void **)&dev_c, sizeof(int) * N * N));
	Error(cudaMalloc((void **)&dev_b, sizeof(int) * N * N));
	Error(cudaMalloc((void **)&dev_a, sizeof(int) * N * N));
	Error(cudaMemcpy(dev_a, a, sizeof(int) * N * N, cudaMemcpyHostToDevice));
	Error(cudaMemcpy(dev_b, b, sizeof(int) * N * N, cudaMemcpyHostToDevice));
	kernel << <block, threads >> > (dev_a, dev_b, dev_c, N);
	Error(cudaGetLastError());
	Error(cudaDeviceSynchronize());
	Error(cudaMemcpy(c, dev_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost));
	std::cout << "Time: " << GetTickCount() - time << std::endl;

	/*std::cout << "C" << std::endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << c[i*N + j] << "  ";
		}
		std::cout << std::endl;
	}*/

	Error(cudaFree(dev_c));
	Error(cudaFree(dev_b));
	Error(cudaFree(dev_a));

	system("pause");
	return 0;

}
