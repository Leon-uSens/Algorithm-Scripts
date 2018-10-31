#include <immintrin.h>

#include "dgemm.h"

void OriginalDgemm(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int i = 0; i < kDimension; ++i)
	{
		for (int j = 0; j < kDimension; ++j)
		{
			// The variable cij is the element c[i][j] in matrix c.
			double cij = 0;

			for (int k = 0; k < kDimension; ++k)
			{
				// c[i][j] = a[i][:] * b[:][j].
				cij += matrix_a[i * kDimension + k] * matrix_b[j + k * kDimension];
			}

			matrix_c[i * kDimension + j] = cij;
		}
	}
}

void DgemmWithAvx(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int i = 0; i < kDimension; ++i)
	{
		for (int j = 0; j < kDimension; j += 4)
		{
			// Load c[i][j : j + 3] to c_avx.
			__m256d c_avx = _mm256_load_pd(&(matrix_c[i * kDimension + j]));

			for (int k = 0; k < kDimension; ++k)
			{
				// Calculate c[i][j : j + 3], which is a[i][:] * b[:][j : j + 3].
				__m256d product_avx = _mm256_mul_pd(_mm256_broadcast_sd(&(matrix_a[i * kDimension + k])),
					_mm256_load_pd(&(matrix_b[k * kDimension + j])));

				c_avx = _mm256_add_pd(c_avx, product_avx);
			}

			// Load c_avx to c[i][j : j + 3].
			_mm256_store_pd(&(matrix_c[i * kDimension + j]), c_avx);
		}
	}
}
