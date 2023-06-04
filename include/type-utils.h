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
  static_assert(!traits<Lhs>::nCols || !traits<Rhs>::nRows ||
                traits<Lhs>::nCols == traits<Rhs>::nRows);
  using type = SparseMatrix<
      traits<Lhs>::nRows ? traits<Lhs>::nRows : 0,
      traits<Rhs>::nCols ? traits<Rhs>::nCols : 0,
      traits<Lhs>::storage == Sparse || traits<Rhs>::storage == Sparse ? Sparse
                                                                       : Dense,
      traits<Lhs>::major>;
};

template <typename Lhs, typename Rhs> struct SumReturnType {
  static_assert((!traits<Lhs>::nRows || !traits<Rhs>::nRows ||
                 traits<Lhs>::nRows == traits<Rhs>::nRows) &&
                (!traits<Lhs>::nCols || !traits<Rhs>::nCols ||
                 traits<Lhs>::nCols == traits<Rhs>::nCols));
  using type = SparseMatrix<
      (traits<Lhs>::nRows | traits<Rhs>::nRows),
      (traits<Lhs>::nCols | traits<Rhs>::nCols),
      traits<Lhs>::storage == Dense || traits<Rhs>::storage == Dense ? Dense
                                                                     : Sparse,
      traits<Lhs>::major>;
};

template <typename Derived> class SparseMatrixBase;

template <typename T> struct IsSparseMatrix {
  static constexpr bool value = false;
};

template <typename Derived> struct IsSparseMatrix<SparseMatrixBase<Derived>> {
  static constexpr bool value = false;
};

template <typename T> using is_spm_v = typename IsSparseMatrix<T>::value;

template <typename T> struct GetDerived {};
template <typename Derived> struct GetDerived<SparseMatrixBase<Derived>> {
  using type = Derived;
};

template <typename T> struct IsStaticVector {
  static constexpr bool value =
      (traits<T>::nRows == 1 || traits<T>::nCols == 1);
};

template <typename T>
using is_static_vector = typename IsStaticVector<T>::value;

} // namespace spmx

// namespace spmx

#endif // SPMX_TYPE_UTILS_H
