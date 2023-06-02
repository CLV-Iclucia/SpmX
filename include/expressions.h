//
// Created by creeper on 23-5-30.
//

#ifndef SPMX_EXPRESSIONS_H
#define SPMX_EXPRESSIONS_H

#include <sparse-matrix.h>
#include <type-utils.h>
namespace spmx {

/**
 * @tparam Lhs
 * @tparam Rhs
 */
template <typename Lhs, typename Rhs>
class LinearExpr : public SparseMatrixBase<LinearExpr<Lhs, Rhs>> {
public:
  using RetType = typename SumReturnType<Lhs, Rhs>::type;
  using Expr = typename LinearExpr<Lhs, Rhs>::type;
  using traits<RetType>::StorageType;
  LinearExpr(Real lhs_coeff, const Lhs &lhs, Real rhs_coeff, const Rhs &rhs)
      : lhs_coeff_(lhs_coeff), rhs_coeff_(rhs_coeff), lhs_(lhs), rhs_(rhs) {}
  class NonZeroIterator;
  RetType Eval() const {
    if (iszero(lhs_coeff_))
      return rhs_;
    if (iszero(rhs_coeff_))
      return lhs_;
    if constexpr (StorageType == Sparse) {
      RetType ret(lhs_.Rows(), lhs_.Cols(), NonZeroEst());
      NonZeroIterator it(*this);
      ret.SetByIterator(it);
      return ret;
    } else if (traits<Lhs>::StorageType == Dense &&
               traits<Rhs>::StorageType == Dense) {
      RetType ret(lhs_.Rows(), lhs_.Cols());
      for (int i = 0; i < OuterDim(); i++)
        for (int j = 0; j < InnerDim(); j++)
          ret.AccessByMajor(i, j) = lhs_coeff_ * lhs_.AccessByMajor(i, j) +
                                    rhs_coeff_ * rhs_.AccessByMajor(i, j);
      return ret;
    } else if (traits<Lhs>::StorageType == Dense) {
      RetType ret(lhs_);
      typename Rhs::NonZeroIterator rhs_it(rhs_);
      while (rhs_it()) {
        ret.AccessByMajor(rhs_it.Outer(), rhs_it.Inner()) =
            lhs_coeff_ * ret.AccessByMajor(rhs_it.Outer(), rhs_it.Inner()) +
            rhs_coeff_ * rhs_it.value();
      }
      return ret;
    } else {
      RetType ret(rhs_);
      typename Lhs::NonZeroIterator lhs_it(lhs_);
      while (lhs_it()) {
        ret.AccessByMajor(lhs_it.Outer(), lhs_it.Inner()) =
            lhs_coeff_ * lhs_it.value() +
            rhs_coeff_ * ret.AccessByMajor(lhs_it.Outer(), lhs_it.Inner());
      }
      return ret;
    }
  }
  Real LhsCoeff() const { return lhs_coeff_; }
  Real RhsCoeff() const { return rhs_coeff_; }
  const Lhs &LhsExpr() const { return lhs_; }
  const Rhs &RhsExpr() const { return rhs_; }
  template <typename RhsExpr>
  LinearExpr<Expr, RhsExpr> operator+(const RhsExpr &rhs) const {
    return LinearExpr<Expr, RhsExpr>(*this, rhs);
  }

  Expr operator*(Real coeff) const {
    return Expr(lhs_coeff_ * coeff, lhs_, rhs_coeff_ * coeff, rhs_);
  }

  friend Expr operator*(Real coeff, const Expr &expr) {
    return Expr(expr.lhs_coeff_ * coeff, expr.lhs_, expr.rhs_coeff_ * coeff,
                expr.rhs_);
  }

  uint OuterDim() const { return lhs_.OuterDim(); }
  uint InnerDim() const { return lhs_.InnerDim(); }
  uint NonZeroEst() const { return lhs_.NonZeroEst() + rhs_.NonZeroEst(); }

private:
  Real lhs_coeff_ = 0.0;
  Real rhs_coeff_ = 0.0;
  const Lhs &lhs_;
  const Rhs &rhs_;
};

template <typename Lhs, typename Rhs>
class LinearExpr<Lhs, Rhs>::NonZeroIterator {
public:
  using Expr = LinearExpr<Lhs, Rhs>;
  explicit NonZeroIterator(const Expr &expr)
      : lhs_it_(expr.LhsExpr()), rhs_it_(expr.RhsExpr()),
        lhs_coeff_(expr.lhs_coeff_), rhs_coeff_(expr.rhs_coeff_) {}
  NonZeroIterator &operator++() {
    if (lhs_it_.Outer() > rhs_it_.Outer())
      rhs_it_++;
    else if (lhs_it_.Outer() < rhs_it_.Outer())
      lhs_it_++;
    else {
      if (lhs_it_.Inner() < rhs_it_.Inner())
        lhs_it_++;
      else if (lhs_it_.Inner() > rhs_it_.Inner())
        rhs_it_++;
      else {
        lhs_it_++;
        rhs_it_++;
      }
    }
    return *this;
  }

  uint Outer() const { return std::min(lhs_it_.Outer(), rhs_it_.Outer()); }
  /**
   * this should only be called when confirmed that lhs_it_.Outer() ==
   * rhs_it_.Outer()
   * @return
   */
  uint Inner() const { return std::min(lhs_it_.Inner(), rhs_it_.Outer()); }
  Real value() const {
    if (lhs_it_.Outer() > rhs_it_.Outer())
      return rhs_it_.value();
    else if (lhs_it_.Outer() < rhs_it_.Outer())
      return lhs_it_.value();
    else {
      if (lhs_it_.Inner() < rhs_it_.Inner())
        return lhs_it_.value();
      else if (lhs_it_.Inner() > rhs_it_.Inner())
        return rhs_it_.value();
      else
        return lhs_coeff_ * lhs_it_.value() + rhs_coeff_ * rhs_it_.value();
    }
  }
  bool operator()() const { return lhs_it_() && rhs_it_(); }

private:
  using LhsIter = typename Lhs::NonZeroIterator;
  using RhsIter = typename Rhs::NonZeroIterator;
  Real lhs_coeff_{};
  Real rhs_coeff_{};
  LhsIter lhs_it_;
  RhsIter rhs_it_;
};

template <typename Lhs, typename Rhs> struct traits<LinearExpr<Lhs, Rhs>> {
  using type = typename SumReturnType<Lhs, Rhs>::type;
  using traits<type>::nRows;
  using traits<type>::nCols;
  using traits<type>::Storage;
  using traits<type>::major;
};

} // namespace spmx
#endif // SPMX_EXPRESSIONS_H
