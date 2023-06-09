//
// Created by creeper on 23-6-3.
//

#ifndef SPMX_SPARSE_MATRIX_BASE_H
#define SPMX_SPARSE_MATRIX_BASE_H

#include <mat-mul-impl.h>
#include <my-stl.h>
#include <spmx-utils.h>
#include <type-utils.h>

namespace spmx {

template <typename Lhs, typename Rhs, int Coeff> class LinearExpr;
template <typename Lhs, typename Rhs> using AddExpr = LinearExpr<Lhs, Rhs, 1>;
template <typename Lhs, typename Rhs> using SubExpr = LinearExpr<Lhs, Rhs, -1>;
template <typename Rhs> class UnaryExpr;
/**
 * I use CRTP for all the matrix and vectors!
 * Use this we can hold sparse matrices in various forms.
 * @tparam Derived
 */
template <typename Derived> class SparseMatrixBase {
public:
  static constexpr uint nRows = traits<Derived>::nRows;
  static constexpr uint nCols = traits<Derived>::nCols;
  static constexpr StorageType storage = traits<Derived>::storage;
  static constexpr StorageMajor major = traits<Derived>::major;
  Derived &derived() { return *(static_cast<Derived *>(this)); }
  const Derived &derived() const {
    return *(static_cast<const Derived *>(this));
  }

  inline typename traits<Derived>::EvalType Eval() const {
    return derived().Eval();
  }
  uint OuterDim() const {
    if constexpr (traits<Derived>::major == RowMajor || traits<Derived>::major == Symmetric)
      return derived().Rows();
    else
      return derived().Cols();
  }
  uint InnerDim() const {
    if constexpr (traits<Derived>::major == RowMajor || traits<Derived>::major == Symmetric)
      return derived().Cols();
    else
      return derived().Rows();
  }
  inline uint OuterIdx(uint i) const { return derived().OuterIdx(i); }
  inline uint InnerIdx(uint i) const { return derived().InnerIdx(i); }
  inline Real Data(uint i) const { return derived().Data(i); }

  inline uint Rows() const { return derived().Rows(); }
  inline uint Cols() const { return derived().Cols(); }
  inline uint NonZeroEst() const { return derived().NonZeroEst(); }
  inline uint Dim() const { return Rows() * Cols(); }

  template <typename Lhs, typename Rhs>
  using ProdRet = typename ProductReturnType<Lhs, Rhs>::type;

  using Lhs = Derived;
  /**
   * Sparse matrix multiplication.
   * All the return types are SparseMatrix, so it makes no sense to support
   * multiplication of other possible matrix storage.
   * @tparam Rhs
   * @param rhs
   * @return
   */
  template <typename Rhs>
  ProdRet<Lhs, Rhs> operator*(const SparseMatrixBase<Rhs> &rhs) const {
    static_assert(traits<Lhs>::storage == traits<Rhs>::storage ||
                      is_supported_vector<Lhs> || is_supported_vector<Rhs>,
                  "Error: Oops, multiplication of sparse matrices and dense "
                  "matrices are not supported yet. SpmX is a small lib "
                  "designed specifically for sparse matrices. More support for "
                  "dense matrices will be provided in future updates");
    if constexpr (is_supported_vector<Lhs> || is_supported_vector<Rhs>) {
      if constexpr (is_supported_vector_of_major_v<Lhs, RowMajor> && is_supported_vector_of_major_v<Rhs, ColMajor>)
        return Dot(*this, rhs);
      if constexpr (is_supported_vector_of_major_v<Rhs, ColMajor>)
        return SpmvImpl(derived(), rhs.derived());
      else if constexpr (is_supported_vector_of_major_v<Lhs, RowMajor>)
        return SpmvImpl(rhs.derived(), derived());
      else if constexpr (is_supported_vector_of_major_v<Lhs, ColMajor> && is_supported_vector_of_major_v<Rhs, RowMajor>)
        return TensorProduct(*this, rhs);
    }
    if constexpr (traits<Lhs>::storage == Dense &&
                  traits<Rhs>::storage == Dense) {
      return DenseDenseMatMulImpl(derived(), rhs.derived());
    } else if constexpr (traits<Lhs>::storage == Dense ||
                         traits<Rhs>::storage == Dense) {
      // TODO: implement this in future updates
    } else {
      if constexpr (traits<Lhs>::major != traits<Rhs>::major)
        return SparseSparseMatMulImpl(derived(), rhs.derived().Transposed());
      return SparseSparseMatMulImpl(derived(), rhs.derived());
    }
  }

  UnaryExpr<Lhs> operator*(Real coeff) const;
  UnaryExpr<Lhs> operator-() const;
  template <typename Rhs> AddExpr<Lhs, Rhs> operator+(const Rhs &rhs) const;
  template <typename Rhs> SubExpr<Lhs, Rhs> operator-(const Rhs &rhs) const;
  template <typename Rhs> Lhs &operator+=(const Rhs &rhs);
  template <typename Rhs> Lhs &operator-=(const Rhs &rhs);
  template <typename Rhs> Lhs &operator*=(const Rhs &rhs);

  Real operator[](uint i) const { return derived().operator[](i); }

  Real &operator[](uint i) { return derived().operator[](i); }

  void toTriplets(std::vector<Triplet> &t_list) const {
    typename Derived::NonZeroIterator it(derived());
    while (it()) {
      t_list.emplace_back(it.Row(), it.Col(), it.value());
      ++it;
    }
  }
};

template <typename Rhs>
UnaryExpr<Rhs> operator*(Real lhs, const SparseMatrixBase<Rhs> &rhs) {
  return UnaryExpr<Rhs>(lhs, rhs.derived());
}

} // namespace spmx

#endif // SPMX_SPARSE_MATRIX_BASE_H
