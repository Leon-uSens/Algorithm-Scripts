#pragma once

#include <vector>

constexpr int kDimension = 160;
constexpr int kSize = kDimension * kDimension;

void OriginalDgemm (const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

void DgemmWithAvx(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

void DgemmWithInstructionLevelParallel (const double* matrix_a, const double* matrix_b, double* matrix_c);

void DgemmWithCacheOptimization (const double* matrix_a, const double* matrix_b, double* matrix_c);

void DgemmWithThreadLevelParallel (const double* matrix_a, const double* matrix_b, double* matrix_c);