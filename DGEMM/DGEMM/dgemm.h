#pragma once

#include <vector>

constexpr int kLength = 640;
constexpr int kSize = kLength * kLength;
constexpr int kUnroll = 4;
constexpr int kBlockLength = 32;

void OriginalDgemm(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

void DgemmWithAvx(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

void DgemmWithInstructionLevelParallel(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

inline void BlockDgemm(int si, int sj, int sk, const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

void DgemmWithCacheOptimization(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);

void DgemmWithThreadLevelParallel(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, std::vector<double>& matrix_c);