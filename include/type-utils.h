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
static constexpr StorageMajor transpose_v = transpose_op<MajorType>::value;

template <typename T> struct traits {};

template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix;

template <typename Lhs, typename Rhs> struct SameShape {
  static constexpr bool value = (!traits<Lhs>::nRows || !traits<Rhs>::nRows ||
                                 traits<Lhs>::nRows == traits<Rhs>::nRows) &&
                                (!traits<Lhs>::nCols || !traits<Rhs>::nCols ||
                                 traits<Lhs>::nCols == traits<Rhs>::nCols);
};

template <typename Lhs, typename Rhs>
static constexpr bool is_same_shape_v = SameShape<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs> struct ProductReturnType {
  static_assert(!traits<Lhs>::nCols || !traits<Rhs>::nRows ||
                traits<Lhs>::nCols == traits<Rhs>::nRows);
  using type = std::conditional_t<
      traits<Lhs>::nRows == 1 && traits<Rhs>::nCols == 1, Real,
      SparseMatrix<traits<Lhs>::nRows, traits<Rhs>::nCols,
                   traits<Lhs>::storage == Dense ||
                           traits<Rhs>::storage == Dense
                       ? Dense
                       : Sparse,
                   traits<Lhs>::major == Symmetric ? traits<Rhs>::major
                                                   : traits<Lhs>::major>>;
};

template <typename Lhs, typename Rhs> struct SumReturnType {
  static_assert(is_same_shape_v<Lhs, Rhs>);
  using type = SparseMatrix<
      (traits<Lhs>::nRows | traits<Rhs>::nRows),
      (traits<Lhs>::nCols | traits<Rhs>::nCols),
      traits<Lhs>::storage == Dense || traits<Rhs>::storage == Dense ? Dense
                                                                     : Sparse,
      traits<Lhs>::major == Symmetric ? traits<Rhs>::major
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

template <typename Derived> class SparseMatrixBase;

template <typename T> struct IsSupportedVector {
  static constexpr bool value = !std::is_same_v<remove_all_t<T>, T> &&
                                IsSupportedVector<remove_all_t<T>>::value;
};

template <uint nRows, uint nCols, StorageType storageType, StorageMajor Major>
struct IsSupportedVector<SparseMatrix<nRows, nCols, storageType, Major>> {
  static constexpr bool value =
      (nRows == 1 && Major == RowMajor) || (nCols == 1 && Major == ColMajor);
};

template <typename Rhs> class UnaryExpr;
template <typename Lhs, typename Rhs, int Coeff> class LinearExpr;
template <typename Rhs> struct IsSupportedVector<UnaryExpr<Rhs>> {
  static constexpr bool value = IsSupportedVector<Rhs>::value;
};

template <typename Lhs, typename Rhs, int Coeff>
struct IsSupportedVector<LinearExpr<Lhs, Rhs, Coeff>> {
  static constexpr bool value = IsSupportedVector<
      typename traits<LinearExpr<Lhs, Rhs, Coeff>>::EvalType>::value;
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

template <typename T, StorageMajor major_> struct is_supported_vector_of_major {
  static constexpr bool value = is_supported_vector<T> &&
                                traits<remove_all_t<T>>::major == major_ &&
                                ((traits<remove_all_t<T>>::major == RowMajor &&
                                  traits<remove_all_t<T>>::nRows == 1) ||
                                 (traits<remove_all_t<T>>::major == ColMajor &&
                                  traits<remove_all_t<T>>::nCols == 1));
};

template <typename T, StorageMajor major_>
static constexpr bool is_supported_vector_of_major_v =
    is_supported_vector_of_major<T, major_>::value;

template <typename Derived> struct IsFixedShape {
  static constexpr bool value = traits<remove_all_t<Derived>>::nRows &&
                                traits<remove_all_t<Derived>>::nCols;
};

template <typename Derived>
static constexpr bool is_fixed_shape_v = IsFixedShape<Derived>::value;
} // namespace spmx

// namespace spmx

#endif // SPMX_TYPE_UTILS_H
