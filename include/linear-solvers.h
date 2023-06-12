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
  const Derived &derived() const { return *static_cast<Derived *>(this); }
  template <typename GeneralMatType>
  Vector<Dense> Solve(const SparseMatrixBase<GeneralMatType> &A,
                      const Vector<Dense> &b) const {
    Vector<Dense> ret(b.Dim());
    derived().Solve(A, b, ret);
    return ret;
  };

protected:
  SolverStatus status_ = Undefined;
};
}// namespace spmx

#endif // SPMX_LINEAR_SOLVERS_H
