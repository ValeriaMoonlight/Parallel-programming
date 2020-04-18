#include <omp.h>
#include <algorithm>
#include <iostream>
#include <Windows.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>
#include <iomanip>

using namespace std;

void randomiseMatrix(int **matrix, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i][j] = rand() % 11;
		}
	}

	return;
}

void outputMatrix(int **matrix, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << matrix[i][j] << "\t";
		}
		cout << endl;
	}

	return;
}

void Check(int **matrix1, int **matrix2, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (matrix1[i][j] != matrix2[i][j])
				cout << "False" << endl;
		}
	}
	cout << "True" << endl;
	return;
}

int **New(int N) {
	int **matrix = new int*[N];
	for (int i = 0; i < N; i++) {
		matrix[i] = new int[N];
	}
	return matrix;
}

void Free(int **matrix, int N) {
	for (int i = 0; i < N; i++) {
		delete[] matrix[i];
	}
	delete matrix;
}

int** matrixMult(int **matrix1, int N, int **matrix2, int **result) {

	unsigned int start_time = clock(); 	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			result[i][j] = 0;
			for (int k = 0; k < N; k++) {
				result[i][j] += (matrix1[i][k] * matrix2[k][j]);
			}
		}
	}
	unsigned int end_time = clock(); 
	unsigned int search_time = end_time - start_time; 
	cout << "Matrix Mult " << search_time << endl;
	return result;
}

int** matrixParalMult(int **matrix1, int N, int **matrix2, int **result) {

	unsigned int start_time = clock(); 
	//int threadsNum = 4;
	omp_set_num_threads(4);
	int i, j, k;
#pragma omp parallel for shared(matrix1, matrix2, result) private(i, j, k)
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result[i][j] = 0;
			for (k = 0; k < N; k++) {
				result[i][j] += (matrix1[i][k] * matrix2[k][j]);
			}
		}
	}

	unsigned int end_time = clock(); 
	unsigned int search_time = end_time - start_time;
	cout << "Matrix Paral Mult " << search_time << endl;

	return result;
}


int main() {
	srand(time(NULL));
	int N = 10;

	int **matrix1;
	int **matrix2;

	matrix1 = New(N);

	matrix2 = New(N);

	randomiseMatrix(matrix1, N);
	randomiseMatrix(matrix2, N);

	int **result = New(N);
	int **result1 = New(N);

	result = matrixMult(matrix1, N, matrix2, result);
	//outputMatrix(result, N);

	result1 = matrixParalMult(matrix1, N, matrix2, result1);
	//outputMatrix(result1, N);

	cout << "DONE" << endl;

	Check(result, result1, N);

	Free(matrix1, N);
	Free(matrix2, N);
	Free(result, N);
	Free(result1, N);

	return 0;
}