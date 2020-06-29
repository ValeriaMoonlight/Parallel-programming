#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <iostream>
#include <ctime>

const int SIZE_OF_BLOCK = 16;

void Error(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void matMult(int *A, int *B, int *C, int N) {

	int A_begin = N * SIZE_OF_BLOCK;
	A_begin = A_begin * blockIdx.y;
	int A_end = A_begin + N - 1;
	int B_begin = SIZE_OF_BLOCK;
	B_begin = B_begin * blockIdx.x;
	int A_step = SIZE_OF_BLOCK;
	int B_step = SIZE_OF_BLOCK * N;
	int sum = 0;
	/*if(blockIdx.x == 1 && blockIdx.y == 1)
	printf("A begin is %d, A end is %d, B begin is %d\n", A_beg, A_end, B_beg);*/
	__shared__ int A_shared[SIZE_OF_BLOCK][SIZE_OF_BLOCK];
	__shared__ int B_shared[SIZE_OF_BLOCK][SIZE_OF_BLOCK];
	for (int i_A = A_begin, i_B = B_begin; i_A <= A_end; i_A += A_step, i_B += B_step) {
		A_shared[threadIdx.y][threadIdx.x] = A[i_A + N * threadIdx.y + threadIdx.x];
		B_shared[threadIdx.y][threadIdx.x] = B[i_B + N * threadIdx.y + threadIdx.x];
		__syncthreads();
		for (int k = 0; k < SIZE_OF_BLOCK; k++) {
			sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
		}
		__syncthreads();
		C[N * SIZE_OF_BLOCK * blockIdx.y + SIZE_OF_BLOCK * blockIdx.x + N * threadIdx.y + threadIdx.x] = sum;
	}
}
int main() {
	srand(time(0));
	int N = 1000;
	dim3 block(N / SIZE_OF_BLOCK, N / SIZE_OF_BLOCK);
	dim3 threads(SIZE_OF_BLOCK, SIZE_OF_BLOCK);
	int *dev_c, *dev_a, *dev_b;
	int *a, *b, *c;
	a = new int[N * N];
	b = new int[N * N];
	c = new int[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i*N + j] = rand()%10;
			b[i*N + j] = rand() % 10;
		}
	}
	/*std::cout << "A" << std::endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << a[i*N + j] << "  ";
		}
		std::cout << std::endl;
	}
	std::cout << "B" << std::endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << b[i*N + j] << "  ";
		}
		std::cout << std::endl;
	}*/
	kernel << <block, threads >> > ();
	int time = GetTickCount();
	Error(cudaMalloc((void **)&dev_c, sizeof(int) * N * N));
	Error(cudaMalloc((void **)&dev_b, sizeof(int) * N * N));
	Error(cudaMalloc((void **)&dev_a, sizeof(int) * N * N));
	Error(cudaMemcpy(dev_a, a, sizeof(int) * N * N, cudaMemcpyHostToDevice));
	Error(cudaMemcpy(dev_b, b, sizeof(int) * N * N, cudaMemcpyHostToDevice));
	//kernel << <block, threads >> > (dev_a, dev_b, dev_c, N);
	matMult << <block, threads >> > (dev_a, dev_b, dev_c, N);
	Error(cudaGetLastError());
	Error(cudaMemcpy(c, dev_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost));
	std::cout << "Time:" << GetTickCount() - time << std::endl;
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
