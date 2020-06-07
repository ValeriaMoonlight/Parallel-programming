#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <iomanip>

void Error(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void kernel(int *G, int count, int k) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	G[i * count + j] = min(G[i * count + j], G[i * count + k] + G[k * count + j]);
}
__global__ void kernel() {
}
int main() {

	int N = 1500;
	srand(time(0));
	int size = 20;
	dim3 block(N / size, N / size);
	dim3 threads(size, size);
	int *dev_G;
	int *G;
	G = new int[N * N];
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++) {
			if (i != j) {
				G[i*N + j] = rand() % 100;
			}
			else
				G[i*N + j] = 0;
		}
	}
	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << G[i*N + j] << "  ";
		}
		std::cout << std::endl;
	}*/
	kernel << <block, threads >> > ();
	Error(cudaGetLastError());
	Error(cudaDeviceSynchronize());
	int time = GetTickCount();
	Error(cudaMalloc((void **)&dev_G, sizeof(int) * N * N));
	Error(cudaMemcpy(dev_G, G, sizeof(int) * N * N, cudaMemcpyHostToDevice));
	for (int i = 0; i < N; i++) {
		kernel << <block, threads >> > (dev_G, N, i);
		Error(cudaGetLastError());
		Error(cudaDeviceSynchronize());
	}
	Error(cudaMemcpy(G, dev_G, sizeof(int) * N * N, cudaMemcpyDeviceToHost));
	printf("%d\n", GetTickCount() - time);
	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << G[i*N + j] << "  ";
		}
		std::cout << std::endl;
	}*/
	Error(cudaFree(dev_G));

	system("pause");
	return 0;

}
