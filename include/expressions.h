//
// Created by creeper on 23-5-30.
//

#ifndef SPMX_EXPRESSIONS_H
#define SPMX_EXPRESSIONS_H

#include <iostream>
#include <sparse-matrix-base.h>
namespace spmx {

template <typename Rhs>
class UnaryExpr : public SparseMatrixBase<UnaryExpr<Rhs>> {
public:
  UnaryExpr(Real coeff, const Rhs &rhs) : coeff_(coeff), rhs_(rhs) {}
  using RetType = Rhs;
  using Expr = UnaryExpr<Rhs>;
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
  friend UnaryExpr<Rhs> operator*(Real lhs, const UnaryExpr<Rhs> &rhs) {
    return UnaryExpr<Rhs>(lhs, rhs);
  }
  class NonZeroIterator {
  public:
    explicit NonZeroIterator(const Expr &expr)
        : coeff_(expr.coeff_), it(expr.rhs_) {}
    bool operator()() const {
      return it();
    }
    NonZeroIterator &operator++() {
      ++it;
      return *this;
    }
    Real Outer() const { return it.Outer(); }
    Real Inner() const { return it.Inner(); }

    Real value() const { return coeff_ * it.value(); }

  private:
    Real coeff_ = 0.0;
    typename Rhs::NonZeroIterator it = nullptr;
  };
  uint NonZeroEst() const {
    return rhs_.NonZeroEst();
  }

  uint Rows() const { return rhs_.Rows(); }
  uint Cols() const { return rhs_.Cols(); }
  uint OuterDim() const { return rhs_.OuterDim(); }
  uint InnerDim() const { return rhs_.InnerDim(); }
private:
  Real coeff_ = 0;
  const Rhs &rhs_ = nullptr;
};

/**
 * @tparam Lhs
 * @tparam Rhs
 */
template <typename Lhs, typename Rhs>
class AddExpr : public SparseMatrixBase<AddExpr<Lhs, Rhs>> {
public:
  using RetType = typename SumReturnType<Lhs, Rhs>::type;
  using Base = SparseMatrixBase<AddExpr<Lhs, Rhs>>;
  using Expr = AddExpr<Lhs, Rhs>;
  using Base::operator+;
  AddExpr(const Lhs &lhs, const Rhs &rhs) : lhs_(lhs), rhs_(rhs) {}

  class NonZeroIterator {
  public:
    explicit NonZeroIterator(const Expr &expr)
        : lhs_it_(expr.LhsExpr()), rhs_it_(expr.RhsExpr()) {}
    NonZeroIterator &operator++() {
      if(!rhs_it_()) {
        ++lhs_it_;
        return *this;
      }
      if(!lhs_it_()) {
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
      if(!rhs_it_()) return lhs_it_.Outer();
      if(!lhs_it_()) return rhs_it_.Outer();
      return std::min(lhs_it_.Outer(), rhs_it_.Outer());
    }
    /**
     * this should only be called when confirmed that lhs_it_.Outer() ==
     * rhs_it_.Outer()
     * @return
     */
    uint Inner() const {
      if(!rhs_it_()) return lhs_it_.Inner();
      if(!lhs_it_()) return rhs_it_.Inner();
      if (lhs_it_.Outer() < rhs_it_.Outer()) return lhs_it_.Inner();
      else if (lhs_it_.Outer() > rhs_it_.Outer()) return rhs_it_.Inner();
      return std::min(lhs_it_.Inner(), rhs_it_.Inner());
    }
    Real value() const {
      if(!rhs_it_()) return lhs_it_.value();
      if(!lhs_it_()) return rhs_it_.value();
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
          return lhs_it_.value() + rhs_it_.value();
      }
    }
    bool operator()() const { return lhs_it_() || rhs_it_(); }

  private:
    using LhsIter = typename Lhs::NonZeroIterator;
    using RhsIter = typename Rhs::NonZeroIterator;
    LhsIter lhs_it_;
    RhsIter rhs_it_;
  };

  RetType Eval() const {
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
              lhs_.AccessByMajor(i, j) + rhs_.AccessByMajor(i, j);
      return ret;
    } else if (traits<Lhs>::StorageType == Dense) {
      RetType ret(lhs_);
      typename Rhs::NonZeroIterator rhs_it(rhs_);
      while (rhs_it()) {
        ret.AccessByMajor(rhs_it.Outer(), rhs_it.Inner()) =
            ret.AccessByMajor(rhs_it.Outer(), rhs_it.Inner()) + rhs_it.value();
      }
      return ret;
    } else {
      RetType ret(rhs_);
      typename Lhs::NonZeroIterator lhs_it(lhs_);
      while (lhs_it()) {
        ret.AccessByMajor(lhs_it.Outer(), lhs_it.Inner()) =
            lhs_it.value() + ret.AccessByMajor(lhs_it.Outer(), lhs_it.Inner());
      }
      return ret;
    }
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
  static constexpr uint nRows = traits<Rhs>::nRows;
  static constexpr uint nCols = traits<Rhs>::nCols;
  static constexpr StorageType storage = traits<Rhs>::storage;
  static constexpr StorageMajor major = traits<Rhs>::major;
};

template <typename Lhs, typename Rhs> struct traits<AddExpr<Lhs, Rhs>> {
  using type = typename SumReturnType<Lhs, Rhs>::type;
  static constexpr uint nRows = traits<type>::nRows;
  static constexpr uint nCols = traits<type>::nCols;
  static constexpr StorageType storage = traits<type>::storage;
  static constexpr StorageMajor major = traits<type>::major;
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

template <typename Derived>
UnaryExpr<typename SparseMatrixBase<Derived>::Lhs>
operator*(Real lhs, const Derived &rhs) {
  return UnaryExpr<Derived>(lhs, rhs);
}

template <typename Derived>
template <typename Rhs>
AddExpr<typename SparseMatrixBase<Derived>::Lhs, Rhs>
SparseMatrixBase<Derived>::operator+(const Rhs &rhs) const {
  return AddExpr<Lhs, Rhs>(derived(), rhs);
}

template <typename Derived>
template <typename Rhs>
AddExpr<typename SparseMatrixBase<Derived>::Lhs, Rhs>
SparseMatrixBase<Derived>::operator-(const Rhs &rhs) const {
  return AddExpr<Lhs, Rhs>(derived(), UnaryExpr<Rhs>(-1, rhs));
}

template <typename Derived>
template <typename Rhs>
typename SparseMatrixBase<Derived>::Lhs &
SparseMatrixBase<Derived>::operator+=(const Rhs &rhs) const {
  derived() = derived() + rhs;
  return derived();
}

template <typename Derived>
template <typename Rhs>
typename SparseMatrixBase<Derived>::Lhs &
SparseMatrixBase<Derived>::operator-=(const Rhs &rhs) const {
  derived() = derived() - rhs;
  return derived();
}

} // namespace spmx
#endif // SPMX_EXPRESSIONS_H
