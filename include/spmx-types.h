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
enum SetOptions { Ordered = 1 << 0, NoRepeat = 1 << 1 };
enum SolverStatus {
  Undefined,
  Success,
  NumericalError,
  NotConverge,
  InvalidInput
};

#define MEMORY_LOG_ALLOC(ClassName, size)                                      \
  std::cout << "******* Memory Log: " << #ClassName << "-allocate " << size    \
            << " *******" << std::endl
#define MEMORY_LOG_DELETE(ClassName, size)                                     \
  std::cout << "******* Memory Log: " << #ClassName << "-delete " << size      \
            << " *******" << std::endl
#define MEMORY_LOG_INVALID_ACCESS(ClassName, idx)                              \
  std::cout << "******* Memory Log: " << #ClassName << "-Invalid access "      \
            << idx << " *******" << std::endl
#define MEMORY_LOG_MESSAGE(ClassName, msg)                                     \
  std::cout << "******* Memory Log: " << #ClassName << "-message " << msg << " *******"       \
            << std::endl
} // namespace spmx
#endif // SPMX_TYPES_H
