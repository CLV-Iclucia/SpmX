//
// Created by creeper on 23-6-3.
//

#ifndef SPMX_SPARSE_MATRIX_BASE_H
#define SPMX_SPARSE_MATRIX_BASE_H

#include <spmx-utils.h>
#include <type-utils.h>

namespace spmx {
template <typename Lhs, typename Rhs> class AddExpr;
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

  inline uint OuterDim() const { return derived().OuterDim(); }
  inline uint InnerDim() const { return derived().InnerDim(); }
  inline uint OuterIdx(uint i) const { return derived().OuterIdx(i); }
  inline uint InnerIdx(uint i) const { return derived().InnerIdx(i); }
  inline Real Data(uint i) const { return derived().Data(i); }

  inline uint Rows() const { return derived().Rows(); }
  inline uint Cols() const { return derived().Cols(); }
  inline uint NonZeroEst() const { return derived().NonZeroEst(); }
  inline uint Dim() const { return Rows() * Cols(); }

  bool IsVector() const { return Rows() == 1 || Cols() == 1; }

  template <typename Lhs, typename Rhs>
  using ProdRet = typename ProductReturnType<Lhs, Rhs>::type;

  template <typename Lhs, typename Rhs>
  using SumRet = typename SumReturnType<Lhs, Rhs>::type;

  using Lhs = Derived;
  template <typename Rhs>
  ProdRet<Lhs, Rhs> operator*(const SparseMatrixBase<Rhs> &rhs) const {
    if constexpr (major != traits<Rhs>::major) {
      uint est_nnz = 0;
      for (uint i = 0; i < OuterDim(); i++) {
        for (uint j = 0; j < rhs.OuterDim(); j++) {
          uint lptr = OuterIdx(i), rptr = rhs.OuterIdx(j);
          while (lptr < InnerIdx(i + 1) && rptr < rhs.InnerIdx(j + 1)) {
            if (InnerIdx(lptr) < rhs.InnerIdx(rptr))
              lptr++;
            else if (rhs.InnerIdx(rptr) < InnerIdx(lptr))
              rptr++;
            else {
              est_nnz++;
              break;
            }
          }
        }
      }
      ProdRet<Lhs, Rhs> ret(Rows(), Cols(), est_nnz);
      uint cnt = 0;
      for (uint i = 0; i < OuterDim(); i++) {
        ret.OuterIdx(i) = cnt;
        for (uint j = 0; j < rhs.OuterDim(); j++) {
          uint lptr = OuterIdx(i), rptr = rhs.OuterIdx(j);
          Real sum = 0.0;
          while (lptr < OuterIdx(i + 1) && rptr < rhs.OuterIdx(j + 1)) {
            if (InnerIdx(lptr) < rhs.InnerIdx(rptr))
              lptr++;
            else if (InnerIdx(lptr) > rhs.InnerIdx(rptr))
              rptr++;
            else {
              sum += Data(lptr) * rhs.Data(rptr);
              lptr++;
              rptr++;
            }
          }
          if (!iszero(sum)) {
            ret.InnerIdx(cnt) = j;
            ret.Data(cnt++) = sum;
          }
        }
        ret.OuterIdx(ret.OuterDim()) = cnt;
      }
      return ret;
    } else {
      uint est_nnz = 0;
//      // estimate the number of non-zero elements of the result
//      BitSet bucket(InnerDim());
//      for (uint i = 0; i < OuterDim(); i++) {
//        bucket.Clear();
//        for (uint j = OuterIdx(i); j < OuterIdx(i + 1); j++) {
//          uint idx = InnerIdx(j);
//          for (uint k = rhs.OuterIdx(idx); k < rhs.OuterIdx(idx + 1); k++)
//            if (!bucket(rhs.InnerIdx(k)))
//              bucket.Set(rhs.InnerIdx(k));
//        }
//        est_nnz += bucket.BitCnt();
//      }
      ProdRet<Lhs, Rhs> ret(Rows(), rhs.Cols(), est_nnz);
//      uint cnt = 0;
//      for (uint i = 0; i < OuterDim(); i++) { // TODO: this is wrong
//        bucket.Clear();
//        ret.OuterIdx(i) = cnt;
//        for (uint j = OuterIdx(i); j < OuterIdx(i + 1); j++) {
//          uint idx = InnerIdx(j);
//          for (uint k = rhs.OuterIdx(idx); k < rhs.OuterIdx(idx + 1); k++) {
//            if (!bucket(rhs.InnerIdx(k))) {
//              ret.inner[ret.nnz++] = rhs.InnerIdx(k);
//            }
//            ret.val[ret.nnz] += Data(j) * rhs.Data(k);
//          }
//        }
//      }
//      ret.OuterIdx(rhs.OuterDim()) = cnt;
      return ret;
    }
  }

  UnaryExpr<Lhs> operator*(Real coeff) const;

  UnaryExpr<Lhs> operator-() const;

  template <typename Rhs> AddExpr<Lhs, Rhs> operator+(const Rhs &rhs) const;

  template <typename Rhs> AddExpr<Lhs, Rhs> operator-(const Rhs &rhs) const;

  template <typename Rhs> Lhs &operator+=(const Rhs &rhs) const;

  template <typename Rhs> Lhs &operator-=(const Rhs &rhs) const;

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

} // namespace spmx

#endif // SPMX_SPARSE_MATRIX_BASE_H
