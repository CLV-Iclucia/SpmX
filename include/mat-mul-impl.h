//
// Created by creeper on 23-6-8.
//

#ifndef SPMX_MAT_MUL_IMPL_H
#define SPMX_MAT_MUL_IMPL_H

#include <cmath>
#include <my-stl.h>
#include <type-utils.h>
namespace spmx {

template <typename Lhs, typename Rhs>
using ProdRet = typename ProductReturnType<Lhs, Rhs>::type;

template <typename Lhs, typename Rhs>
inline ProdRet<Lhs, Rhs> DenseDenseMatMulImpl(const Lhs &lhsEval,
                                              const Rhs &rhsEval) {
  static_assert(is_spm_v<Lhs> && is_spm_v<Rhs> &&
                traits<Lhs>::storage == Sparse &&
                traits<Rhs>::storage == Sparse &&
                traits<Lhs>::major == traits<Rhs>::major);
  ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
  // TODO: optimize this!
  for (int i = 0; i < lhsEval.OuterDim(); i++) {
    for (int j = 0; j < rhsEval.InnerDim(); j++) {
      for (int k = 0; k < lhsEval.InnerDim(); k++) {
        ret(i, j) += lhsEval(i, k) * rhsEval(k, j);
      }
    }
  }
  return ret;
}

template <typename LhsInnerIterator, typename RhsInnerIterator>
inline Real SparseSparseVectorDotKernel(LhsInnerIterator &lhs_it,
                              RhsInnerIterator &rhs_it) {
  Real sum = 0.0;
  while (lhs_it() && rhs_it()) {
    if (lhs_it.Inner() < rhs_it.Inner())
      ++lhs_it;
    else if (lhs_it.Inner() > rhs_it.Inner())
      ++rhs_it;
    else {
      sum += lhs_it.value() * rhs_it.value();
      ++lhs_it;
      ++rhs_it;
    }
  }
  return sum;
}

/**
 * Here we process matrix * vector and ignores the whether the vector is col-vector or row-vector
 * So the cases are
 * Matrix(Sparse/Dense, Col/Row) * Vector(Sparse/Dense)
 * for vector we always call its NonZeroIterator
 * So we can focus one the latter
 * @tparam Lhs
 * @tparam Rhs
 * @param lhsEval
 * @param rhsEval
 * @return
 */
template <typename Lhs, typename Rhs>
inline ProdRet<Lhs, Rhs> SpmvImpl(const Lhs &lhsEval, const Rhs &rhsEval) {
  static_assert(is_supported_vector<Lhs> || is_supported_vector<Rhs>);
  constexpr uint option = ((traits<Lhs>::storage == Sparse) << 2) |
                          ((traits<Lhs>::major == RowMajor) << 1) |
                          (traits<Rhs>::storage == Dense);
  if constexpr (traits<Lhs>::storage == Sparse) {
    if constexpr (traits<Lhs>::major == RowMajor) {
      if constexpr (traits<Rhs>::storage == Dense) {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        for (uint i = 0; i < lhsEval.OuterDim(); i++) {
          for (typename Lhs::InnerIterator it(lhsEval, i); it(); ++it) {
            ret(i) += it.value() * rhsEval(it.Inner());
          }
        }
        return ret;
      } else {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols(), rhsEval.Cols());
        for (uint i = 0; i < lhsEval.OuterDim(); i++) {
          typename Lhs::InnerIterator lhs_it(lhsEval, i);
          typename Rhs::NonZeroIterator rhs_it(rhsEval);
          Real sum = SparseSparseVectorDotKernel(lhs_it, rhs_it);
          uint nnz = 0;
          if (!iszero(sum)) {
            ret.Innert(nnz) = i;
            ret.Data(nnz++) = sum;
          }
        }
        return ret;
      }
    } else {
      if constexpr (traits<Rhs>::storage == Dense) {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        for (uint i = 0; i < lhsEval.OuterDim(); i++) {
          for (typename Lhs::InnerIterator it(lhsEval, i); it(); ++it) {
            ret(it.Inner()) += it.value() * rhsEval(it.Inner());
          }
        }
        return ret;
      } else {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        Array<int> mask(rhsEval.Cols());
        Array<int> idx_bucket(rhsEval.Cols());
        Array<Real> extended(rhsEval.Cols());
        mask.Fill(0);
        extended.Fill(0);
        uint nnz = 0;
        for (typename Rhs::NonZeroIterator rhs_it(rhsEval); rhs_it();
             ++rhs_it) {
          for (typename Lhs::InnerIterator lhs_it(lhsEval, rhs_it.Inner());
               lhs_it(); ++lhs_it) {
            extended(lhs_it.Inner()) += lhs_it.value() * rhs_it.value();
            if (!mask(lhs_it.Inner())) {
              mask(lhs_it.Inner()) = 1;
              idx_bucket[nnz++] = lhs_it.Inner();
            }
          }
        }
        std::sort(idx_bucket.Data(), idx_bucket.Data() + nnz);
        for (uint j = 0; j < nnz; j++) {
          ret.InnerIdx(j) = idx_bucket[j];
          ret.Data(j) = extended[idx_bucket[j]];
        }
      }
    }
  } else {
    if constexpr (traits<Lhs>::major == RowMajor) {
      if constexpr (traits<Rhs>::storage == Dense) {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        for (uint i = 0; i < lhsEval.OuterDim(); i++) {
          for (uint j = 0; j < lhsEval.InnerDim(); j++) {
            ret(i) += lhsEval.AccessByMajor(i, j) * rhsEval(j);
          }
        }
        return ret;
      } else {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        for (uint i = 0; i < lhsEval.OuterDim(); i++) {
          for (typename Rhs::NonZeroIterator it(rhsEval); it(); ++it) {
            ret(i) += lhsEval.AccessByMajor(i, it.Inner()) * it.value();
          }
        }
        return ret;
      }
    } else {
      if constexpr (traits<Rhs>::storage == Dense) {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        for (uint i = 0; i < lhsEval.OuterDim(); i++) {
          for (uint j = 0; j < lhsEval.InnerDim(); j++) {
            ret.AccessByMajor(j) += lhsEval.AccessByMajor(i, j) * rhsEval(i);
          }
        }
        return ret;
      } else {
        ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols());
        for (typename Rhs::NonZeroIterator rhs_it(rhsEval); rhs_it();
             ++rhs_it) {
          for (typename Rhs::InnerIterator lhs_it(lhsEval, rhs_it.Inner());
               lhs_it(); ++lhs_it) {
            ret(lhs_it.Inner()) += rhs_it.value() * lhs_it.value();
          }
        }
        return ret;
      }
    }
  }
}

template <typename Lhs, typename Rhs>
inline ProdRet<Lhs, Rhs> SparseSparseMatMulImpl(const Lhs &lhsEval,
                                                const Rhs &rhsEval) {
  static_assert(traits<Lhs>::storage == Sparse &&
                traits<Rhs>::storage == Sparse &&
                traits<Lhs>::major == traits<Rhs>::major);
  ProdRet<Lhs, Rhs> ret(lhsEval.Rows(), rhsEval.Cols(),
                        lhsEval.NonZeroEst() + rhsEval.NonZeroEst());
  Array<int> mask(rhsEval.Cols());
  Array<int> idx_bucket(rhsEval.Cols());
  Array<Real> extended(rhsEval.Cols());
  ret.OuterIdx(0) = 0;
  // estimate the number of non-zero elements of the result
  for (uint i = 0; i < lhsEval.OuterDim(); i++) {
    uint nnz = 0;
    // 1. calc the result of an inner vector in extended form
    for (typename Lhs::InnerIterator lhs_it(lhsEval, i); lhs_it(); ++lhs_it) {
      for (typename Rhs::InnerIterator rhs_it(rhsEval, lhs_it.Inner());
           rhs_it(); ++rhs_it) {
        extended(rhs_it.Inner()) += lhs_it.value() * rhs_it.value();
        if (!mask(rhs_it.Inner())) {
          idx_bucket[nnz++] = rhs_it.Inner();
          mask(rhs_it.Inner()) = 1;
        }
      }
    }
    // idx_bucket[0:nnz] contains all the inner idx in outer(i)
    // inner_idx_[outer_idx[i]:outer_idx[i+1]], now we need to sort them
    // 2. write back
    std::sort(idx_bucket.Data(), idx_bucket.Data() + nnz);
    for (uint j = 0; j < nnz; j++) {
      ret.InnerIdx(ret.OuterIdx(i) + j) = idx_bucket[j];
      ret.Data(ret.OuterIdx(i) + j) = extended[idx_bucket[j]];
      extended[idx_bucket[j]] = 0;
      mask[idx_bucket[j]] = 0;
    }
    ret.OuterIdx(i + 1) = ret.OuterIdx(i) + nnz;
  }
  ret.Prune();
  return ret;
}

} // namespace spmx
#endif // SPMX_MAT_MUL_IMPL_H
