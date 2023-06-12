//
// Created by creeper on 23-5-19.
//

#ifndef SPMX_ITERATIVE_SOLVERS_H
#define SPMX_ITERATIVE_SOLVERS_H

#include <sparse-matrix.h>
#include <spmx-utils.h>
#include <expressions.h>
namespace spmx {

// TODO: classical iterative solvers
template <typename Derived> class FactorizeSolver {
public:
  using typename traits<Derived>::MatDerived;
  SolverStatus info() const { return status_; }
  Derived *derived() const { return static_cast<Derived *>(this); }
  void Compute(const SparseMatrixBase<MatDerived> &A) {
    derived()->ComputeImpl(A);
  };

  template <StorageType VecStorage>
  void Solve(const SparseMatrixBase<MatDerived> &A, const Vector<VecStorage> &b,
             Vector<Dense> &ret) const {
    derived()->SolveImpl(A, b, ret);
  };

  template <StorageType VecStorage>
  Vector<Dense> Solve(const SparseMatrixBase<MatDerived> &A,
                      const Vector<VecStorage> &b) const {
    Vector<Dense> ret(b.Dim());
    derived()->SolveImpl(A, b, ret);
    return ret;
  };

protected:
  SolverStatus status_ = Undefined;
};

template <typename Derived> class IterativeSolverBase {
public:
  using traits<Derived>::MatDerived;
  SolverStatus info() const { return status_; }
  void SetMaxRounds(int max_rounds) { max_rounds_ = max_rounds; }
  void SetPrecision(Real eps) { eps_ = eps; }
  uint MaxRounds() const { return max_rounds_; }
  Real Precision() const { return eps_; }
  uint Rounds() const { return total_rounds_; }

protected:
  SolverStatus status_ = Undefined;
  int max_rounds_ = -1; ///< -1 stands for iterate until convergence
  uint total_rounds_ = 0;
  Real eps_ = 1e-10;
};

template <typename Derived>
class JacobiSolver final : public IterativeSolverBase<JacobiSolver<Derived>> {
public:
  template <StorageType VecStorage>
  void SolveImpl(const SparseMatrixBase<Derived> &A,
                 const Vector<VecStorage> &b, Vector<Dense> &x) const {
    RandFill(x);
  }
};

template <typename Derived>
class GaussSeidelSolver final
    : public IterativeSolverBase<GaussSeidelSolver<Derived>> {
public:
  template <StorageType VecStorage>
  void SolveImpl(const SparseMatrixBase<Derived> &A,
                 const Vector<VecStorage> &b, Vector<Dense> &ret) const {}
};

template <typename Derived>
class SorSolver final : public IterativeSolverBase<SorSolver<Derived>> {
public:
  template <StorageType VecStorage>
  void SolveImpl(const SparseMatrixBase<Derived> &A,
                 const Vector<VecStorage> &b, Vector<Dense> &ret) const {}
  Real Omega() const { return omega_; }
  void SetOmega(Real omega) { omega_ = omega; }

private:
  Real omega_ = 1.0;
};

template <typename Derived, typename Preconditioner>
class ConjugateGradientSolver final
    : public IterativeSolverBase<ConjugateGradientSolver<Derived, Preconditioner>> {
public:
  template <typename MatDerived, StorageType VecStorage>
  void SolveImpl(const SparseMatrixBase<MatDerived> &A,
                 const Vector<VecStorage> &b, Vector<Dense> &x) const {
    RandFill(x);
    if constexpr (std::is_same_v<Preconditioner, void>) {
      Vector<Dense> r(b - A * x);
      Vector<Dense> p(r);
      Vector<Dense> Ap(A * p);
      Real r_norm_sqr = L2NormSqr(r);
      for (total_rounds_ = 1; max_rounds_ < 0 || total_rounds_ < max_rounds_;
           total_rounds_++) {
        Real alpha = r_norm_sqr / Dot(p, Ap);
        x += p * alpha;
        if (std::abs(alpha) * L1Norm(p) < eps_)
          return;
        r -= alpha * Ap;
        Real beta = L2NormSqr(r) / r_norm_sqr;
        p = r + beta * p;
        r_norm_sqr = L2NormSqr(r);
        Ap = A * p;
      }
      status_ = NotConverge;
    } else {
      // TODO: Add precondition
    }
  }

private:
  using type = ConjugateGradientSolver<Derived, Preconditioner>;
  using Base = IterativeSolverBase<type>;
  using Base::max_rounds_;
  using Base::status_;
  using Base::total_rounds_;
  using Base::eps_;
};

template<typename Derived_, typename Precond_> struct traits<ConjugateGradientSolver<Derived_, Precond_>> {
  using MatDerived = Derived_;
};
template<typename Derived_> struct traits<JacobiSolver<Derived_>> {
  using MatDerived = Derived_;
};

template<typename Derived_> struct traits<GaussSeidelSolver<Derived_>> {
  using MatDerived = Derived_;
};
template<typename Derived_> struct traits<SorSolver<Derived_>> {
  using MatDerived = Derived_;
};
} // namespace spmx
#endif // SPMX_ITERATIVE_SOLVERS_H
