//
// Created by creeper on 23-5-16.
//

#ifndef SPMX_SPMX_CHOLESKY_H
#define SPMX_SPMX_CHOLESKY_H

#include "elimination-tree.h"
#include <linear-solvers.h>
#include <sparse-matrix.h>

namespace spmx {

template <typename Derived>
class SparseCholeskyBase : public FactorizeSolver<SparseCholeskyBase<Derived>> {
public:
  virtual void Analyse(const SparseMatrix &A) {
    static_cast<Derived *>(this)->Analyse(A);
  }
  virtual void Factorize() { static_cast<Derived *>(this)->Factorize(); }
  void Compute(const SparseMatrix &A) {
    Analyse(A);
    Factorize(A);
  }
  template <VectorStorage VecStorage>
  Vector<Dense> solve(const SparseMatrix &A, const Vector<VecStorage> &b) {}
};

class SimplicialLLT : SparseCholeskyBase<SimplicialLLT> {
private:
  EliminationTree etree_;
  SparseMatrix L;
public:
  SimplicialLLT() = default;
  void Analyse(const SparseMatrix &A) override {
    if (!A.IsSquare()) {
      std::cerr
          << "Error: Input matrix for Cholesky solver must be a square matrix"
          << std::endl;
      status_ = InvalidInput;
      return ;
    }
    etree_.BuildFromMatrix(A);

  }
};
} // namespace spmx

#endif // SPMX_SPMX_CHOLESKY_H
