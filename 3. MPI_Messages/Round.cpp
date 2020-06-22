#include <iostream>
#include <string>
#include <mpi.h>

using namespace std;

void MPI_Round() {
	int rank, total;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Status status[2];
	int value[10], input[10];

	if (rank == 0) {
		for (int i = 0; i < 10; ++i) {
			value[i] = i;
		}
		MPI_Send(&value, 10, MPI_INT, (rank + 1) % total, 1, MPI_COMM_WORLD);
	}
	MPI_Recv(&input, 10, MPI_INT, (total + rank - 1) % total, 1, MPI_COMM_WORLD, &status[0]);

	if (rank != 0) {
		MPI_Send(&input, 10, MPI_INT, (rank + 1) % total, 1, MPI_COMM_WORLD);
	}
	cout << "rank " << rank << " recived: ";
	for (int i = 0; i < 10; ++i) {
		cout << input[i] << "\t";
	}
	cout << endl;
}

int main(int argc, char** argv) {

	MPI_Init(NULL, NULL);
	MPI_Round();
	MPI_Finalize();
	return 0;
}
