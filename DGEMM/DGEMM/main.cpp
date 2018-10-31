#include <iostream>
#include <ctime>
#include <Windows.h>

#include "dgemm.h"

int main(int argc, char** argv)
{
	/*---------------------------------------------------------------------------------------------------*/
	std::vector<double> matrix_a(kSize, 1);
	std::vector<double> matrix_b(kSize, 2);
	std::vector<double> matrix_c(kSize, 0);
	
	std::clock_t start_time;
	std::clock_t end_time;
	/*---------------------------------------------------------------------------------------------------*/
	std::cout << "Calculating original DGEMM..." << std::endl;
	start_time = std::clock();
	OriginalDgemm(matrix_a, matrix_b, matrix_c);
	end_time = std::clock();
	long time_1 = end_time - start_time;
	std::cout << "Original DGEMM takes: " << time_1 << std::endl;

	std::cout << "Calculating DGEMM with AVX..." << std::endl;
	start_time = std::clock();
	DgemmWithAvx(matrix_a, matrix_b, matrix_c);
	end_time = std::clock();
	long time_2 = end_time - start_time;
	std::cout << "DGEMM with AVX takes: " << time_2 << std::endl;

	std::cout << "Calculating DGEMM with AVX and instruction-level parallel..." << std::endl;
	start_time = std::clock();
	DgemmWithInstructionLevelParallel(matrix_a, matrix_b, matrix_c);
	end_time = std::clock();
	long time_3 = end_time - start_time;
	std::cout << "DGEMM with AVX and instruction-level parallel: " << time_3 << std::endl;

	std::cout << "Calculating DGEMM with AVX, instruction-level parallel and cache optimization..." << std::endl;
	start_time = std::clock();
	DgemmWithCacheOptimization(matrix_a, matrix_b, matrix_c);
	end_time = std::clock();
	long time_4 = end_time - start_time;
	std::cout << "DGEMM with AVX, instruction-level parallel and cache optimization takes: " << time_4 << std::endl;

	std::cout << "Calculating DGEMM with AVX, instruction-level parallel, cache optimization and thread-level parallel..." << std::endl;
	start_time = std::clock();
	DgemmWithThreadLevelParallel(matrix_a, matrix_b, matrix_c);
	end_time = std::clock();
	long time_5 = end_time - start_time;
	std::cout << "DGEMM with AVX, instruction-level parallel, cache optimization and thread-level parallel takes: " << time_5 << std::endl;
	
	getchar();
	/*---------------------------------------------------------------------------------------------------*/
	return 0;
}