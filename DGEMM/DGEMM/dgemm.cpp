#include <immintrin.h>

#include "dgemm.h"

void OriginalDgemm(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int i = 0; i < kLength; ++i)
	{
		for (int j = 0; j < kLength; ++j)
		{
			// The variable cij is the element c[i][j] in matrix c.
			double cij = 0;

			for (int k = 0; k < kLength; ++k)
			{
				// c[i][j] = a[i][:] * b[:][j].
				cij += matrix_a[i * kLength + k] * matrix_b[j + k * kLength];
			}

			matrix_c[i * kLength + j] = cij;
		}
	}
}

void DgemmWithAvx(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int i = 0; i < kLength; ++i)
	{
		// Avx 256, which is 4 times of doube (64 bits).
		for (int j = 0; j < kLength; j += 4)
		{
			// Load c[i][j : j + 3] to c_avx.
			__m256d c_avx = _mm256_load_pd(&(matrix_c[i * kLength + j]));

			for (int k = 0; k < kLength; ++k)
			{
				// Calculate c[i][j : j + 3], which is a[i][:] * b[:][j : j + 3].
				__m256d product_avx = _mm256_mul_pd(_mm256_broadcast_sd(&(matrix_a[i * kLength + k])),
					_mm256_load_pd(&(matrix_b[k * kLength + j])));

				c_avx = _mm256_add_pd(c_avx, product_avx);
			}

			// Load c_avx to c[i][j : j + 3].
			_mm256_store_pd(&(matrix_c[i * kLength + j]), c_avx);
		}
	}
}

void DgemmWithInstructionLevelParallel(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int i = 0; i < kLength; ++i)
	{
		// Unroll the for loop by kUnroll (if kLength is much bigger than 4 * kUnroll).
		for (int j = 0; j < kLength; j += kUnroll * 4)
		{
			__m256d c_avx[kUnroll];
			for (int m = 0; m < kUnroll; ++m)
			{
				c_avx[m] = _mm256_load_pd(&(matrix_c[i * kLength + j + m * 4]));
			}

			for (int k = 0; k < kLength; ++k)
			{
				__m256d a_avx = _mm256_broadcast_sd(&(matrix_a[i * kLength + k]));

				for (int m = 0; m < kUnroll; ++m)
				{
					__m256d product_avx = _mm256_mul_pd(a_avx, _mm256_load_pd(&(matrix_b[k * kLength + j + m * 4])));
					c_avx[m] = _mm256_add_pd(c_avx[m], product_avx);
				}
			}

			for (int m = 0; m < kUnroll; ++m)
			{
				_mm256_store_pd(&(matrix_c[i * kLength + j + m * 4]), c_avx[m]);
			}
		}
	}
}

inline void BlockDgemm(int si, int sj, int sk, const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int i = si; i < si + kBlockLength; ++i)
	{
		// Unroll the for loop by kUnroll (if kLength is much bigger than 4 * kUnroll).
		for (int j = sj; j < sj + kBlockLength; j += kUnroll * 4)
		{
			__m256d c_avx[kUnroll];
			for (int m = 0; m < kUnroll; ++m)
			{
				c_avx[m] = _mm256_load_pd(&(matrix_c[i * kLength + j + m * 4]));
			}

			for (int k = sk; k < sk + kBlockLength; ++k)
			{
				__m256d a_avx = _mm256_broadcast_sd(&(matrix_a[i * kLength + k]));

				for (int m = 0; m < kUnroll; ++m)
				{
					__m256d product_avx = _mm256_mul_pd(a_avx, _mm256_load_pd(&(matrix_b[k * kLength + j + m * 4])));
					c_avx[m] = _mm256_add_pd(c_avx[m], product_avx);
				}
			}

			for (int m = 0; m < kUnroll; ++m)
			{
				_mm256_store_pd(&(matrix_c[i * kLength + j + m * 4]), c_avx[m]);
			}
		}
	}
}

void DgemmWithCacheOptimization(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	for (int si = 0; si < kLength; si += kBlockLength)
	{
		for (int sj = 0; sj < kLength; sj += kBlockLength)
		{
			for (int sk = 0; sk < kLength; sk += kBlockLength)
			{
				BlockDgemm(si, sj, sk, matrix_a, matrix_b, matrix_c);
			}
		}
	}
}

void DgemmWithThreadLevelParallel(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c)
{
	// Thread-level parallel.
#pragma omp parallel for
	for (int si = 0; si < kLength; si += kBlockLength)
	{
		for (int sj = 0; sj < kLength; sj += kBlockLength)
		{
			for (int sk = 0; sk < kLength; sk += kBlockLength)
			{
				BlockDgemm(si, sj, sk, matrix_a, matrix_b, matrix_c);
			}
		}
	}
}
