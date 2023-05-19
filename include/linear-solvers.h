//
// Created by creeper on 23-3-5.
//

#ifndef SPMX_LINEAR_SOLVERS_H
#define SPMX_LINEAR_SOLVERS_H

#include <sparse-matrix.h>
#include <spmx-types.h>

namespace spmx {

template <typename Derived> class FactorizeSolver {
public:
  SolverStatus info() const { return status_; }
  virtual void compute(const SparseMatrix &A) {
    static_cast<Derived *>(this)->compute(A);
  };

  template <VectorStorage VecStorage>
  void solve(const Vector<VecStorage> &b, Vector<Dense> &ret) const {
    static_cast<Derived *>(this)->solve(b, ret);
  };

  template <VectorStorage VecStorage>
  Vector<Dense> solve(const Vector<VecStorage> &b) const {
    Vector<Dense> ret(b.dim());
    static_cast<Derived *>(this)->solve(b, ret);
    return ret;
  };

protected:
  SolverStatus status_ = Undefined;
};

template <typename Derived> class IterativeSolverBase {
public:
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

class JacobiSolver final : public IterativeSolverBase<JacobiSolver> {
public:
  template <VectorStorage VecStorage>
  void solve(const SparseMatrix &A, const Vector<VecStorage> &b,
             Vector<Dense> &ret) const {}
};

class GaussSeidelSolver final : public IterativeSolverBase<GaussSeidelSolver> {
public:
  template <VectorStorage VecStorage>
  void solve(const SparseMatrix &A, const Vector<VecStorage> &b,
             Vector<Dense> &ret) const {}
};

template <class Preconditioner>
class CGSolver final : public IterativeSolverBase<CGSolver<Preconditioner> > {
public:

  template <VectorStorage VecStorage>
  void solve(const SparseMatrix &A, const Vector<VecStorage> &b,
             Vector<Dense> &x) const {
    x.RandFill();
    Vector<Dense> r(b - A * x);
    Vector<Dense> p(r);
    Vector<Dense> Ap(A * p);
    Real r_norm_sqr = r.L2NormSqr();
    for (total_rounds_ = 1; max_rounds_ < 0 || (total_rounds_ < max_rounds_); total_rounds_++) {
      Real alpha = r_norm_sqr / p.dot(Ap);
      x.saxpy(p, alpha);
      if(std::abs(alpha) * p.L1Norm() < eps_) return;
      r.saxpy(Ap, -alpha);
      Real beta = r.L2NormSqr() / r_norm_sqr;
      p.scadd(r, beta);
      r_norm_sqr = r.L2NormSqr();
      Ap = A * p;
    }
    status_ = NotConverge;
  }
};

} // namespace spmx

#endif // SPMX_LINEAR_SOLVERS_H
