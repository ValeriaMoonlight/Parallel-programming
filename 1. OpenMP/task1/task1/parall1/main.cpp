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

int main()
{
#pragma omp parallel num_threads(3)
	{
		int rank = omp_get_thread_num();
		//srand(static_cast<unsigned int>(time(0))^rank-1);
		//int num = rand()%97-rank;
		random_device rd;
		mt19937 mersenne(rd()^rank);
		int num = mersenne()%997;
	#pragma omp critical
		{
			ofstream fout("rank.txt", ios_base::app);
			if (fout.is_open()) {
				fout << "rank " << rank << " num " << num << endl;
				cout << "rank " << rank << " num " << num << endl;
			}
			fout.close();
		}
	}
	return 0;
}