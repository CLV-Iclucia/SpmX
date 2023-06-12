//
// Created by creeper on 23-5-30.
//

#ifndef SPMX_EXPRESSIONS_H
#define SPMX_EXPRESSIONS_H

#include <algorithm>
#include <iostream>
#include <sparse-matrix-base.h>
namespace spmx {

/**
 * @note Any expression class used below requires the sparse matrices to be in
 * the same order! So when wrapping them in operator overloading, make sure that
 * the matrices are properly transposed.
 */

/**
 * UnaryExpr implement a nested InnerIterator by directly using the
 * InnerIterator of Rhs. So, if Rhs doesn't have a nested InnerIterator, neither
 * does UnaryExpr.
 * @tparam Rhs
 */
template <typename Rhs>
class UnaryExpr : public SparseMatrixBase<UnaryExpr<Rhs>> {
public:
  UnaryExpr(Real coeff, const Rhs &rhs) : coeff_(coeff), rhs_(rhs) {}
  using RetType = Rhs;
  using Base = SparseMatrixBase<UnaryExpr<Rhs>>;
  using Expr = UnaryExpr<Rhs>;
  using Base::operator*;
  using Base::operator+;
  using Base::operator-;
  Rhs Eval() {
    if constexpr (traits<Rhs>::storage == Dense) {
      RetType ret(rhs_.Rows(), rhs_.Cols());
      for (uint i = 0; i < rhs_.Outer(); i++)
        for (uint j = 0; j < rhs_.Inner(); j++)
          ret.AccessByMajor(i, j) = coeff_ * rhs_.AccessByMajor(i, j);
      return ret;
    } else {
      typename Rhs::NonZeroIterator it(rhs_);
      RetType ret(rhs_.Rows(), rhs_.Cols, rhs_.NonZeroEst());
      ret.SetByIterator(it);
      return ret;
    }
  }

  bool IsSquare() const { return rhs_.IsSquare(); }
  class InnerIterator {
  public:
    explicit InnerIterator(const Expr &expr, uint i)
        : coeff_(expr.coeff_), it(expr.rhs_, i) {}
    bool operator()() const { return it(); }
    [[maybe_unused]] InnerIterator &operator++() {
      ++it;
      return *this;
    }
    uint Inner() const { return it.Inner(); }
    Real value() const { return coeff_ * it.value(); }

  private:
    Real coeff_ = 0;
    typename Rhs::InnerIterator it;
  };

  class NonZeroIterator {
  public:
    explicit NonZeroIterator(const Expr &expr)
        : coeff_(expr.coeff_), it(expr.rhs_) {}
    bool operator()() const { return it(); }
    [[maybe_unused]] NonZeroIterator &operator++() {
      ++it;
      return *this;
    }
    uint Outer() const { return it.Outer(); }
    uint Inner() const { return it.Inner(); }
    Real value() const { return coeff_ * it.value(); }

  private:
    Real coeff_ = 0.0;
    typename Rhs::NonZeroIterator it = nullptr;
  };

  Real operator()(uint i, uint j) const {
    static_assert(traits<RetType>::storage == Dense);
    return coeff_ * rhs_(i, j);
  }
  Real AccessByMajor(uint i, uint j) const {
    static_assert(traits<RetType>::storage == Dense);
    return coeff_ * rhs_.AcessByMajor(i, j);
  }
  uint NonZeroEst() const { return rhs_.NonZeroEst(); }
  uint Rows() const { return rhs_.Rows(); }
  uint Cols() const { return rhs_.Cols(); }
  uint OuterDim() const { return rhs_.OuterDim(); }
  uint InnerDim() const { return rhs_.InnerDim(); }

private:
  Real coeff_ = 0;
  const Rhs &rhs_ = nullptr;
};

/**
 * @note LinearExpr requires both to be in the same storage
 * @tparam Lhs
 * @tparam Rhs
 */
template <typename Lhs, typename Rhs, int Coeff>
class LinearExpr : public SparseMatrixBase<LinearExpr<Lhs, Rhs, Coeff>> {
  static_assert(traits<Lhs>::storage == traits<Rhs>::storage,
                "Error: current version doesn't support implicit addition of "
                "dense matrices and sparse matrices.");
public:
  using RetType = typename SumReturnType<Lhs, Rhs>::type;
  using Base = SparseMatrixBase<LinearExpr<Lhs, Rhs, Coeff>>;
  using Expr = LinearExpr<Lhs, Rhs, Coeff>;
  using Base::operator*;
  using Base::operator+;
  using Base::operator-;
  LinearExpr(const Lhs &lhs, const Rhs &rhs) : lhs_(lhs), rhs_(rhs) {}

  class NonZeroIterator {
  public:
    explicit NonZeroIterator(const Expr &expr)
        : lhs_it_(expr.LhsExpr()), rhs_it_(expr.RhsExpr()) {}
    [[maybe_unused]] NonZeroIterator &operator++() {
      if (!rhs_it_()) {
        ++lhs_it_;
        return *this;
      }
      if (!lhs_it_()) {
        ++rhs_it_;
        return *this;
      }
      if (lhs_it_.Outer() > rhs_it_.Outer())
        ++rhs_it_;
      else if (lhs_it_.Outer() < rhs_it_.Outer())
        ++lhs_it_;
      else {
        if (lhs_it_.Inner() < rhs_it_.Inner())
          ++lhs_it_;
        else if (lhs_it_.Inner() > rhs_it_.Inner())
          ++rhs_it_;
        else {
          ++lhs_it_;
          ++rhs_it_;
        }
      }
      return *this;
    }

    uint Outer() const {
      if (!rhs_it_())
        return lhs_it_.Outer();
      if (!lhs_it_())
        return rhs_it_.Outer();
      return std::min(lhs_it_.Outer(), rhs_it_.Outer());
    }
    /**
     * this should only be called when confirmed that lhs_it_.Outer() ==
     * rhs_it_.Outer()
     * @return
     */
    uint Inner() const {
      if (!rhs_it_())
        return lhs_it_.Inner();
      if (!lhs_it_())
        return rhs_it_.Inner();
      if (lhs_it_.Outer() < rhs_it_.Outer())
        return lhs_it_.Inner();
      else if (lhs_it_.Outer() > rhs_it_.Outer())
        return rhs_it_.Inner();
      return std::min(lhs_it_.Inner(), rhs_it_.Inner());
    }
    Real value() const {
      if (!rhs_it_())
        return lhs_it_.value();
      if (!lhs_it_())
        return Coeff * rhs_it_.value();
      if (lhs_it_.Outer() > rhs_it_.Outer())
        return Coeff * rhs_it_.value();
      else if (lhs_it_.Outer() < rhs_it_.Outer())
        return lhs_it_.value();
      else {
        if (lhs_it_.Inner() < rhs_it_.Inner())
          return lhs_it_.value();
        else if (lhs_it_.Inner() > rhs_it_.Inner())
          return Coeff * rhs_it_.value();
        else
          return lhs_it_.value() + Coeff * rhs_it_.value();
      }
    }
    bool operator()() const { return lhs_it_() || rhs_it_(); }

  private:
    using LhsIter = typename Lhs::NonZeroIterator;
    using RhsIter = typename Rhs::NonZeroIterator;
    LhsIter lhs_it_;
    RhsIter rhs_it_;
  };

  bool IsSquare() const { return rhs_.IsSquare(); }
  RetType Eval() const {
    static_assert(
        traits<Lhs>::storage == traits<Rhs>::storage,
        "Error: current version doesn't support implicit linear combination "
        "of a dense matrix and a sparse matrix since it's difficult to "
        "generate compile-time logic. For now please use an explicit cast to "
        "do the same thing.\n SpmX is a small lib designed specifically for "
        "sparse matrices. More support for dense matrices will be provided in "
        "future updates");
    if constexpr (traits<RetType>::storage == Sparse) {
      RetType ret(lhs_.Rows(), lhs_.Cols(), NonZeroEst());
      NonZeroIterator it(*this);
      ret.SetByIterator(it);
      return ret;
    } else if (traits<Lhs>::StorageType == Dense &&
               traits<Rhs>::StorageType == Dense) {
      // TODO: Obviously, this can be optimized
      RetType ret(lhs_.Rows(), lhs_.Cols());
      for (int i = 0; i < OuterDim(); i++)
        for (int j = 0; j < InnerDim(); j++)
          ret.AccessByMajor(i, j) =
              lhs_.AccessByMajor(i, j) + Coeff * rhs_.AccessByMajor(i, j);
      return ret;
    } /* else if (traits<Lhs>::StorageType == Dense) {
       RetType ret(lhs_.Eval());
       typename Rhs::NonZeroIterator rhs_it(rhs_);
       while (rhs_it()) {
         ret.AccessByMajor(rhs_it.Outer(), rhs_it.Inner()) =
             ret.AccessByMajor(rhs_it.Outer(), rhs_it.Inner()) + rhs_it.value();
       }
       return ret;
     } else {
       RetType ret(rhs_.Eval());
       typename Lhs::NonZeroIterator lhs_it(lhs_);
       while (lhs_it()) {
         ret.AccessByMajor(lhs_it.Outer(), lhs_it.Inner()) =
             lhs_it.value() + ret.AccessByMajor(lhs_it.Outer(), lhs_it.Inner());
       }
       return ret;
     }*/
  }
  class InnerIterator {
  public:
    explicit InnerIterator(const Expr &expr, uint i)
        : lhs_it_(expr.lhs_, i), rhs_it_(expr.rhs_, i) {}
    bool operator()() const { return lhs_it_() && rhs_it_(); }
    [[maybe_unused]] InnerIterator &operator++() {
      if (!rhs_it_()) {
        ++lhs_it_;
        return *this;
      }
      if (!lhs_it_()) {
        ++rhs_it_;
        return *this;
      }
      if (lhs_it_.Inner() < rhs_it_.Inner())
        ++lhs_it_;
      else if (lhs_it_.Inner() > rhs_it_.Inner())
        ++rhs_it_;
      else {
        ++lhs_it_;
        ++rhs_it_;
      }
      return *this;
    }
    uint Inner() const {
      if (!lhs_it_())
        return rhs_it_.Inner();
      if (!rhs_it_())
        return lhs_it_.Inner();
      return std::min(lhs_it_.Inner(), rhs_it_.Inner());
    }
    Real value() const {
      if (!rhs_it_())
        return lhs_it_.value();
      if (!lhs_it_())
        return Coeff * rhs_it_.value();
      if (lhs_it_.Inner() < rhs_it_.Inner())
        return lhs_it_.value();
      else if (lhs_it_.Inner() > rhs_it_.Inner())
        return Coeff * rhs_it_.value();
      else
        return lhs_it_.value() + Coeff * rhs_it_.value();
    }

  private:
    using LhsIter = typename Lhs::InnerIterator;
    using RhsIter = typename Rhs::InnerIterator;
    LhsIter lhs_it_;
    RhsIter rhs_it_;
  };
  Real operator()(uint i, uint j) const {
    static_assert(traits<RetType>::storage == Dense);
    return lhs_(i, j) + Coeff * rhs_(i, j);
  }
  Real AccessByMajor(uint i, uint j) const {
    static_assert(traits<RetType>::storage == Dense);
    return lhs_.AccessByMajor(i, j) + Coeff * rhs_.AcessByMajor(i, j);
  }
  const Lhs &LhsExpr() const { return lhs_; }
  const Rhs &RhsExpr() const { return rhs_; }
  uint Rows() const { return lhs_.Rows(); }
  uint Cols() const { return lhs_.Cols(); }
  uint OuterDim() const { return lhs_.OuterDim(); }
  uint InnerDim() const { return lhs_.InnerDim(); }
  uint NonZeroEst() const { return lhs_.NonZeroEst() + rhs_.NonZeroEst(); }

private:
  const Lhs &lhs_;
  const Rhs &rhs_;
};

template <typename Rhs> struct traits<UnaryExpr<Rhs>> {
  using EvalType = Rhs;
  static constexpr uint nRows = traits<Rhs>::nRows;
  static constexpr uint nCols = traits<Rhs>::nCols;
  static constexpr StorageType storage = traits<Rhs>::storage;
  static constexpr StorageMajor major = traits<Rhs>::major;
};

template <typename Lhs, typename Rhs, int Coeff>
struct traits<LinearExpr<Lhs, Rhs, Coeff>> {
  using EvalType = typename SumReturnType<Lhs, Rhs>::type;
  static constexpr uint nRows = traits<EvalType>::nRows;
  static constexpr uint nCols = traits<EvalType>::nCols;
  static constexpr StorageType storage = traits<EvalType>::storage;
  static constexpr StorageMajor major = traits<EvalType>::major;
};

template <typename Derived>
UnaryExpr<typename SparseMatrixBase<Derived>::Lhs>
SparseMatrixBase<Derived>::operator*(Real lhs) const {
  return UnaryExpr<Derived>(lhs, derived());
}

template <typename Derived>
UnaryExpr<typename SparseMatrixBase<Derived>::Lhs>
SparseMatrixBase<Derived>::operator-() const {
  return UnaryExpr<Derived>(-1, derived());
}

template <typename Lhs, typename Rhs> using AddExpr = LinearExpr<Lhs, Rhs, 1>;

template <typename Lhs, typename Rhs> using SubExpr = LinearExpr<Lhs, Rhs, -1>;

template <typename Derived>
template <typename Rhs>
AddExpr<typename SparseMatrixBase<Derived>::Lhs, Rhs>
SparseMatrixBase<Derived>::operator+(const Rhs &rhs) const {

  return AddExpr<Lhs, Rhs>(derived(), rhs);
}

template <typename Derived>
template <typename Rhs>
SubExpr<typename SparseMatrixBase<Derived>::Lhs, Rhs>
SparseMatrixBase<Derived>::operator-(const Rhs &rhs) const {
  return SubExpr<Lhs, Rhs>(derived(), rhs);
}

template <typename Derived>
template <typename Rhs>
typename SparseMatrixBase<Derived>::Lhs &
SparseMatrixBase<Derived>::operator+=(const Rhs &rhs) {
  static_assert(
      !(traits<Derived>::storage == Sparse && traits<Rhs>::storage == Dense),
      "Error: current version doesn't support assigning a dense matrix to a "
      "sparse matrix. SpmX is a small lib designed specifically for sparse "
      "matrices. More support for dense matrices will be provided in future "
      "updates");
  if (traits<Derived>::storage == Sparse) {
    derived() = derived() + rhs;
  } else {
    if constexpr (traits<Rhs>::storage == Dense) {

    } else {
    }
  }
  return derived();
}

template <typename Derived>
template <typename Rhs>
typename SparseMatrixBase<Derived>::Lhs &
SparseMatrixBase<Derived>::operator-=(const Rhs &rhs) {
  if (traits<Derived>::storage == Sparse) {
    derived() = derived() - rhs;
  } else {
    if constexpr (traits<Rhs>::storage == Dense) {

    } else {
    }
  }
  return derived();
}

template <typename Derived>
template <typename Rhs>
typename SparseMatrixBase<Derived>::Lhs &
SparseMatrixBase<Derived>::operator*=(const Rhs &rhs) {
  return derived() = derived() * rhs;
}

} // namespace spmx
#endif // SPMX_EXPRESSIONS_H
