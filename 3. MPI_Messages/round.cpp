#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <string>
#include <mpi.h>

int main(int argc, char** argv) {
	int mSize = 10;
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	int rank, total;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Request Req[2];
	MPI_Status Stat[2];
	
	int* value = new int[mSize];
	for (int i = 0; i < mSize; ++i) {
		value[i] = i;
	}
	
	int* input = new int[mSize];
	if (rank == 0) {
		MPI_Send(value, mSize, MPI_INT, (rank + 1) % total, 1, MPI_COMM_WORLD);
	}
	MPI_Recv(input, mSize, MPI_INT, (total + rank - 1) % total, 1, MPI_COMM_WORLD, &Stat[0]);
	if (rank != 0) {
		MPI_Send(input, mSize, MPI_INT, (rank + 1) % total, 1, MPI_COMM_WORLD);
	}
	
	std::cout << "Rank " << rank << " recived:" << std::endl;
	for (int i = 0; i < mSize; ++i) {
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;

	// Finalize the MPI environment.
	MPI_Finalize();
}