//
// Created by creeper on 23-2-23.
//

#ifndef SPMX_TYPES_H
#define SPMX_TYPES_H
#include <tuple>
namespace spmx {
using Real = double;
using uint = unsigned int;
using std::size_t;
using Triplet = std::tuple<uint, uint, Real>;
static const double DOUBLE_EPS = 1e-14;
static const float FLOAT_EPS = 1e-6;
enum StorageType { Sparse, Dense };
enum StorageMajor { RowMajor, ColMajor };
enum SolverStatus {
  Undefined,
  Success,
  NumericalError,
  NotConverge,
  InvalidInput
};
} // namespace spmx
#endif // SPMX_TYPES_H
