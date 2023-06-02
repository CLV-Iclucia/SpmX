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
  template <StorageMajor Type> void ICompute(const SparseMatrix<Type> &A) {
    static_cast<Derived *>(this)->Compute(A);
  };

  template <StorageMajor Type, StorageType VecStorage>
  void Solve(const SparseMatrix<Type> &A, const Vector<VecStorage> &b,
             Vector<Dense> &ret) const {
    static_cast<Derived *>(this)->SolveImpl(A, b, ret);
  };

  template <StorageMajor Type, StorageType VecStorage>
  Vector<Dense> Solve(const SparseMatrix<Type> &A,
                      const Vector<VecStorage> &b) const {
    Vector<Dense> ret(b.Dim());
    static_cast<Derived *>(this)->SolveImpl(A, b, ret);
    return ret;
  };

protected:
  SolverStatus status_ = Undefined;
};

// TODO: TriangularSolver
class TriangularSolver {
public:
  template <StorageMajor Type>
  void Solve(const SparseMatrix<Type> &L, const Vector<Sparse>& b, Vector<Dense>& x) {

  }
  template <StorageMajor Type>
  void Solve(const SparseMatrix<Type> &L, const Vector<Dense>& b, Vector<Dense>& x) {

  }
  template<StorageMajor Type, StorageType VecStorage>
  Vector<Dense> Solve(const SparseMatrix<Type> &L, const Vector<VecStorage>& b) {
    Vector<Dense> x(L.Cols());
    Solve(L, b, x);
    return x;
  }
  template<StorageMajor Type, StorageType VecStorage>
  Vector<Dense> Solve(const SparseMatrix<Type> &L, const Vector<Dense>& b) {
    Vector<Dense> x(L.Cols());
    Solve(L, b, x);
    return x;
  }
  SolverStatus info() const { return status_; }
private:
  SolverStatus status_;
};

} // namespace spmx

#endif // SPMX_LINEAR_SOLVERS_H
