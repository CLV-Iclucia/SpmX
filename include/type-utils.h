//
// Created by creeper on 23-5-19.
//

#ifndef SPMX_TYPE_UTILS_H
#define SPMX_TYPE_UTILS_H

#include <spmx-types.h>
#include <type_traits>

namespace spmx {

template <StorageMajor MajorType> struct transpose_op {
  static constexpr StorageMajor value =
      (MajorType == RowMajor) ? ColMajor : RowMajor;
};

template <StorageMajor MajorType>
using transpose_t = typename transpose_op<MajorType>::type;

template <uint i> struct compile_time_uint {
  using type = std::conditional_t<i != 0, const uint, uint>;
};

template <uint i>
using compile_time_uint_t = typename compile_time_uint<i>::type;

template <typename T> struct traits {};

template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix;

template <typename Lhs, typename Rhs> struct ProductReturnType {
  static_assert(!Lhs::nCols || !Rhs::nRows || Lhs::nCols == Rhs::nRows);
  using type =
      SparseMatrix<Lhs::nRows ? Lhs::nRows : 0, Rhs::nCols ? Rhs::nCols : 0,
                   Lhs::storage == Sparse || Rhs::storage == Sparse ? Sparse
                                                                    : Dense,
                   Lhs::major>;
};

template <typename Lhs, typename Rhs> struct SumReturnType {
  static_assert((!Lhs::nRows || !Rhs::nRows || Lhs::nRows == Rhs::nRows) &&
                (!Lhs::nCols || !Rhs::nCols || Lhs::nCols == Rhs::nCols));
  using type =
      SparseMatrix<(Lhs::nRows || Rhs::nRows) ? (Lhs::nRows | Rhs::nRows) : 0,
                   (Lhs::nCols || Rhs::nCols) ? (Lhs::nCols | Rhs::nCols) : 0,
                   Lhs::storage == Dense || Rhs::storage == Dense ? Dense
                                                                  : Sparse,
                   Lhs::major>;
};

} // namespace spmx

// namespace spmx

#endif // SPMX_TYPE_UTILS_H
