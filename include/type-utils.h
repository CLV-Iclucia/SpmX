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
      MajorType == Symmetric ? Symmetric
                             : (MajorType == RowMajor ? ColMajor : RowMajor);
};

template <StorageMajor MajorType>
using transpose_t = typename transpose_op<MajorType>::type;

template <typename T> struct traits {};

template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix;

template <typename Lhs, typename Rhs> struct ProductReturnType {
  static_assert(!traits<Lhs>::nCols || !traits<Rhs>::nRows ||
                traits<Lhs>::nCols == traits<Rhs>::nRows);
  using type = SparseMatrix<
      traits<Lhs>::nRows, traits<Rhs>::nCols,
      traits<Lhs>::storage == Dense || traits<Rhs>::storage == Dense ? Dense
                                                                     : Sparse,
      traits<Lhs>::major == Symmetric && traits<Lhs>::major == Symmetric
          ? Symmetric
          : traits<Lhs>::major>;
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
      traits<Lhs>::major == Symmetric && traits<Lhs>::major == Symmetric
          ? Symmetric
          : traits<Lhs>::major>;
};

template <typename T> struct remove_all {
  using type = std::remove_const_t<std::remove_reference_t<T>>;
};

template <typename T> using remove_all_t = typename remove_all<T>::type;

template <typename Derived> class SparseMatrixBase;

template <typename T> struct IsSparseMatrix {
  static constexpr bool value = !std::is_same_v<remove_all_t<T>, T> &&
                                IsSparseMatrix<remove_all_t<T>>::value;
};

template <uint nRows, uint nCols, StorageType storageType, StorageMajor Major>
struct IsSparseMatrix<SparseMatrix<nRows, nCols, storageType, Major>> {
  static constexpr bool value = true;
};

template <typename T> static constexpr bool is_spm_v = IsSparseMatrix<T>::value;

template <typename T> struct GetDerived {};
template <typename Derived> struct GetDerived<SparseMatrixBase<Derived>> {
  using type = Derived;
};

template <typename T> struct IsSupportedVector {
  static constexpr bool value = !std::is_same_v<remove_all_t<T>, T> &&
                                IsSupportedVector<remove_all_t<T>>::value;
};

template <uint nRows, uint nCols, StorageType storageType, StorageMajor Major>
struct IsSupportedVector<SparseMatrix<nRows, nCols, storageType, Major>> {
  static constexpr bool value =
      (nRows == 1 && Major == RowMajor) || (nCols == 1 && Major == ColMajor);
};

template <typename T>
static constexpr bool is_supported_vector = IsSupportedVector<T>::value;

template <bool Cond, typename T> struct FinalizedWhen {
  using type = std::conditional_t<Cond, const T, T>;
};

template <bool Cond, typename T>
using finalized_when_t = typename FinalizedWhen<Cond, T>::type;

template <typename Lhs, typename Rhs> struct SameMajor {
  static constexpr bool value = traits<Lhs>::major == Symmetric ||
                                traits<Rhs>::major == Symmetric ||
                                traits<Lhs>::major == traits<Rhs>::major;
};

template <typename Lhs, typename Rhs>
static constexpr bool is_same_major = SameMajor<Lhs, Rhs>::value;

} // namespace spmx

// namespace spmx

#endif // SPMX_TYPE_UTILS_H
