#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <mpi.h>
#include <queue>

int main(int argc, char** argv) {
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	int rank, total;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	if (rank == 0) {
		
		srand(time(0));
		std::vector<int> values;
		for (int i = 0; i < 10; ++i) {
			values.push_back(rand() % 10);
		}
		std::cout << "Array: ";
		for (int i = 0; i < 10; ++i) {
			std::cout << values[i] << " ";
		}
		std::cout << std::endl;
		std::vector<int> results;
		int current_pos = 0;
		int current_rank = 1;
		int flag = 1;
		do {
			
			for (int i = 1; i < total; ++i) {
				
				if (current_pos == values.size())
				{
					values.clear();
					break;
				}
				
				if (current_pos == values.size() - 1)
				{
					results.push_back(values[current_pos]);
					values.clear();
					current_pos = 0;
					break;
				}
				
				int nums[2];
				nums[0] = values[current_pos];
				nums[1] = values[(size_t)current_pos + 1];
				MPI_Send(&flag, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
				MPI_Send(&nums, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
				std::cout << "Numbers " << nums[0] << " " << nums[1] << " send to " << i << " rank" << std::endl;
				current_rank = i + 1;
				current_pos += 2;
			}
			
			for (int i = 1; i < current_rank; ++i)
			{
				MPI_Status status;
				int val;
				MPI_Recv(&val, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
				std::cout << "From " << i << " recieved number " << val << std::endl;
				results.push_back(val);
			}
			
			if (results.size() == 1) {
				std::cout << "Result: " << results[0] << std::endl;
				break;
			}
			
			else
			{
				
				if (values.size() == 0 || values.size() == 1)
				{
					values = results;
					results.clear();
					current_pos = 0;
				}
			}
		} while (true);
		flag = 0;
		
		for (int i = 1; i < total; ++i)
		{
			MPI_Send(&flag, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}
	}
	else {
		int input[2];
		int result;
		int have_value;
		MPI_Status status;
		while (true) {
			
			MPI_Recv(&have_value, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
			if (!have_value)
				break;
			MPI_Recv(&input, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
			result = input[0] + input[1];
			MPI_Send(&result, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

		}
	}
	MPI_Finalize();
}
