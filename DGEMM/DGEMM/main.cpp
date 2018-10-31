#include <iostream>
#include <Windows.h>

#include "dgemm.h"

int main(int argc, char** argv)
{
	std::vector<double> matrix_a(kSize, 1);
	std::vector<double> matrix_b(kSize, 2);

	/*std::vector<double> matrix_a{ 1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4 };
	std::vector<double> matrix_b{ 5,6,7,8,5,6,7,8,5,6,7,8,5,6,7,8 };*/

	std::vector<double> matrix_c(kSize, 0);

	//OriginalDgemm(matrix_a, matrix_b, matrix_c);
	DgemmWithAvx(matrix_a, matrix_b, matrix_c);

	int element_counter{ 0 };

	while (element_counter < kSize)
	{
		std::cout << matrix_c[element_counter] << std::endl;

		element_counter++;
	}

	getchar();

	return 0;
}