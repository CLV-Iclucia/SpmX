//
// Created by creeper on 23-5-19.
//

#ifndef SPMX_ITERATIVE_SOLVERS_H
#define SPMX_ITERATIVE_SOLVERS_H

#include <csignal>
#include <expressions.h>
#include <functional>
#include <parallel.h>
#include <sparse-matrix.h>
#include <spmx-utils.h>

namespace spmx {

// TODO: classical iterative solvers
template <typename Derived> class IterativeSolverBase {
public:
  SolverStatus info() const { return status_; }
  void SetMaxRounds(int max_rounds) { max_rounds_ = max_rounds; }
  void SetPrecision(Real eps) { eps_ = eps; }
  uint MaxRounds() const { return max_rounds_; }
  Real Precision() const { return eps_; }
  uint Rounds() const { return total_rounds_; }
  const Derived &derived() const { return *static_cast<Derived *>(this); }
  Derived &derived() { return *static_cast<Derived *>(this); };
  template <typename GeneralMatType>
  Vector<Dense> Solve(const GeneralMatType &A, const Vector<Dense> &b) {
    Vector<Dense> ret(b.Dim());
    derived().Solve(A, b, ret);
    return ret;
  }

protected:
  SolverStatus status_ = Undefined;
  int max_rounds_ = -1; ///< -1 stands for iterate until convergence
  uint total_rounds_ = 0;
  Real eps_ = 1e-5;
};

template <typename MatType>
class JacobiSolver final : public IterativeSolverBase<JacobiSolver<MatType>> {
public:
  using Base = IterativeSolverBase<JacobiSolver<MatType>>;
  static constexpr uint Size = traits<MatType>::nRows;
  template <StorageType VecStorage>
  void Solve(const MatType &A, const Vector<VecStorage> &b,
             Vector<Dense> &x) const {
    static_assert(is_spm_v<MatType>);
    if (A.Rows() != A.Cols()) {
      status_ = InvalidInput;
      return ;
    }

  }
private:
  using Base::status_;
};

template <typename Derived>
class GaussSeidelSolver final
    : public IterativeSolverBase<GaussSeidelSolver<Derived>> {
public:
  template <StorageType VecStorage>
  void Solve(const SparseMatrixBase<Derived> &A, const Vector<VecStorage> &b,
             Vector<Dense> &ret) const {}
};

template <typename Derived>
class SorSolver final : public IterativeSolverBase<SorSolver<Derived>> {
public:
  template <StorageType VecStorage>
  void Solve(const SparseMatrixBase<Derived> &A, const Vector<VecStorage> &b,
             Vector<Dense> &ret) const {}
  Real Omega() const { return omega_; }
  void SetOmega(Real omega) { omega_ = omega; }

private:
  Real omega_ = 1.0;
};

template <typename MatType, typename Preconditioner = void>
class ConjugateGradientSolver final
    : public IterativeSolverBase<
          ConjugateGradientSolver<MatType, Preconditioner>> {
  static_assert(traits<MatType>::major == Symmetric &&
                    traits<MatType>::storage == Sparse,
                "Error: to use the sparse cholesky factorization solver, the "
                "matrix has to be declared explicitly as sparse and symmetric");
  static constexpr uint Size = traits<MatType>::nRows | traits<MatType>::nCols;
  using Base =
      IterativeSolverBase<ConjugateGradientSolver<MatType, Preconditioner>>;

public:
  using Base::SetMaxRounds;
  using Base::SetPrecision;
  using Base::Solve;

  void Solve(const MatType &A, const Vector<Dense, Size> &b,
             Vector<Dense, Size> &x) {
    if constexpr (std::is_same_v<Preconditioner, void>) {
      Vector<Dense, Size> r(b - A * x);
      Vector<Dense, Size> p(r);
      Vector<Dense, Size> Ap(A * p);
      Real r_norm_sqr = L2NormSqr(r);
      using SpmvKer = decltype(spmv_ker);
      int tasks = (b.Dim() + BLOCK_SIZE - 1) / BLOCK_SIZE;
      ParallelExecuter<SpmvKer, const MatType &, const Vector<Dense, Size> &,
                       Vector<Dense, Size> &>
          parallel_spmv(
              spmv_ker,
              std::min(static_cast<int>(std::thread::hardware_concurrency()),
                       std::min(NUM_MAX_THREADS, tasks)),
              A, p, Ap);
      bool worth_parallel =
          (tasks >= PARALLEL_THRESHOLD_DIM &&
           A.NonZeroEst() >= PARALLEL_THRESHOLD_NNZ * A.Rows());
      for (total_rounds_ = 1;
           max_rounds_ < 0 || static_cast<int>(total_rounds_) < max_rounds_;
           total_rounds_++) {
        Real pAp = Dot(p, Ap);
        [[unlikely]] if (iszero(pAp)) {
          status_ = Success;
          return;
        }
        Real alpha = r_norm_sqr / pAp;
        x += alpha * p;
        r -= alpha * Ap;
        Real new_r_norm_sqr = L2NormSqr(r);
        if (new_r_norm_sqr < eps_ * eps_) {
          status_ = Success;
          return;
        }
        Real beta = new_r_norm_sqr / r_norm_sqr;
        p = r + beta * p;
        r_norm_sqr = new_r_norm_sqr;
        if (worth_parallel)
          parallel_spmv.run(tasks);
        else
          Ap = A * p;
      }
      status_ = NotConverge;
    } else {
      // TODO: Add precondition
    }
  }

private:
  using Base::eps_;
  using Base::max_rounds_;
  using Base::status_;
  using Base::total_rounds_;
  ParallelSpmvKernel<MatType, Vector<Dense, Size>> spmv_ker;
};

template <typename Derived_, typename Precond_>
struct traits<ConjugateGradientSolver<Derived_, Precond_>> {
  using MatDerived = Derived_;
};
template <typename Derived_> struct traits<JacobiSolver<Derived_>> {
  using MatDerived = Derived_;
};

template <typename Derived_> struct traits<GaussSeidelSolver<Derived_>> {
  using MatDerived = Derived_;
};
template <typename Derived_> struct traits<SorSolver<Derived_>> {
  using MatDerived = Derived_;
};

} // namespace spmx
#endif // SPMX_ITERATIVE_SOLVERS_H
