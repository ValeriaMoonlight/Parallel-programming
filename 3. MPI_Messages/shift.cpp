#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <mpi.h>
#include <string>

int main(int argc, char** argv) {
	// Initialize the MPI environment
	int mSize = 10;
	MPI_Init(NULL, NULL);
	int rank, total;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Request Req[2];
	MPI_Status Stat[2];
	int* value = new int[mSize];
	int* input = new int[mSize];

	if (rank == 0) {
		srand((rank + 2) * 444);
		for (int i = 0; i < mSize; ++i) {
			value[i] = rand() % 10;
		}
		MPI_Isend(value, mSize, MPI_INT, (rank + 1) % total, 1, MPI_COMM_WORLD, &Req[0]);
		MPI_Wait(&Req[0], &Stat[0]);
	}

	MPI_Irecv(input, mSize, MPI_INT, (total + rank - 1) % total, 1, MPI_COMM_WORLD, &Req[0]);
	MPI_Wait(&Req[0], &Stat[0]);
	if (rank != 0) {
		srand((rank + 2) * 333);
		for (int i = 0; i < mSize; ++i) {
			value[i] = rand() % 10;
		}
		MPI_Isend(value, mSize, MPI_INT, (rank + 1) % total, 1, MPI_COMM_WORLD, &Req[0]);
		MPI_Wait(&Req[0], &Stat[0]);
	}

	std::cout << "Rank " << rank << " generated:" << std::endl;
	for (int i = 0; i < mSize; ++i) {
		std::cout << value[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "Rank " << rank << " recived:" << std::endl;
	for (int i = 0; i < mSize; ++i)
		std::cout << input[i] << " ";
	std::cout << std::endl;

	// Finalize the MPI environment.
	MPI_Finalize();
}