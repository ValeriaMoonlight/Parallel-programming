#include <iostream>
#include <ctime>
#include <mpi.h>
#include <omp.h>
using namespace std;


int main(int argc, char** argv) {

	MPI_Init(NULL, NULL);
	int seed = (int)time(0);
	srand(time(0));
	int rank_proc, rank_thread, total;
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_proc);

#pragma omp parallel num_threads(4)
	{
#pragma omp critical 
		{
			rank_thread = omp_get_thread_num();
			seed += rank_proc * rand()*rand();
			srand(seed);
			int num = rand();
			cout << "I am " << rank_thread << " thread from " << rank_proc << " process my random number = " << num << endl;
		}
	}
	MPI_Finalize();
}
