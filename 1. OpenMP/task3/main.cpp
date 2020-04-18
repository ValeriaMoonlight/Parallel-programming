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

void OutputMatrix(int **matrix, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << matrix[i][j] << "\t";
		}
		cout << endl;
	}
	return;
}

void RandomData(int **matrix, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j) 
				matrix[i][j] = -1;
			else if (j<i)
				matrix[i][j] = matrix[j][i] = rand() % 11;
		}
	}

	//OutputMatrix(matrix, N);
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

int Min(int A, int B) { 
	int Result = (A < B) ? A : B; 
	if ((A < 0) && (B >= 0)) Result = B; 
	if ((B < 0) && (A >= 0)) Result = A; 
	if ((A < 0) && (B < 0)) Result = -1; 
	return Result; 
}


int** ParallelFloyd(int **pMatrix, int N) { 
	unsigned int start_time = clock(); 	
	//int threadsNum = 4;
	omp_set_num_threads(4);
	for(int k = 0; k < N; k++) 
#pragma omp parallel for
		for(int i = 0; i < N; i++) 
			for(int j = 0; j < N; j++) 
				if((pMatrix[i][k] != -1) && (pMatrix[k][j] != -1)) { 
					pMatrix[i][j] = Min(pMatrix[i][j], pMatrix[i][k] + pMatrix[k][j]);
				} 
	
	unsigned int end_time = clock(); 
	unsigned int search_time = end_time - start_time;
	cout << "Parallel Floyd " << search_time << endl;
	
	return pMatrix;
	//OutputMatrix(pMatrix, N);
}

int** Floyd(int **pMatrix, int N) {
	unsigned int start_time = clock(); 

	for (int k = 0; k < N; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				if ((pMatrix[i][k] != -1) && (pMatrix[k][j] != -1))
					pMatrix[i][j] = Min(pMatrix[i][j], pMatrix[i][k] + pMatrix[k][j]);

	unsigned int end_time = clock(); 
	unsigned int search_time = end_time - start_time; 
	cout << "Floyd " << search_time << endl;

	return pMatrix;
	//OutputMatrix(pMatrix, N);
}


int main() {
	srand(time(NULL));
	int **pMatrix;  
	int **pMatrix1; 
	int N = 5000;  
		
	pMatrix=New(N);
	pMatrix1=New(N);

	RandomData(pMatrix, N);
	pMatrix1 = pMatrix;
	
	int **res;
	int **res1;
	res=New(N);
	res1=New(N);

	res=ParallelFloyd(pMatrix, N); 
	res1=Floyd(pMatrix1, N);
	
	cout << "DONE" << endl;

	Check(res, res1, N);
	
	Free(pMatrix, N);
	Free(pMatrix1, N);
	Free(res, N);
	Free(res1, N);
	return 0;
}