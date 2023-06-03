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
  using typename traits<Derived>::MatDerived;
  Derived *derived() const { return static_cast<Derived *>(this); }
  void Analyse(const SparseMatrixBase<MatDerived> &A) {
    static_cast<Derived *>(this)->AnalyseImpl(A);
  }
  void Factorize() { static_cast<Derived *>(this)->FactorizeImpl(); }
  void ComputeImpl(const SparseMatrixBase<MatDerived> &A) {
    Analyse(A);
    Factorize(A);
  }
  template <StorageType VecStorage>
  Vector<Dense> SolveImpl(const Vector<VecStorage> &b) {}

protected:
  using FactorizeSolver<SparseCholeskyBase<Derived>>::status_;
};

template <typename Derived, bool LDLT>
class SimplicialCholesky
    : SparseCholeskyBase<SimplicialCholesky<Derived, LDLT>> {
private:
  EliminationTree* etree_ = nullptr;
public:
  SimplicialCholesky() {
    etree_ = new EliminationTree;
  }
  void AnalyseImpl(const SparseMatrixBase<Derived> &A) {
    if (!A.IsSquare()) {
      std::cerr
          << "Error: Input matrix for Cholesky solver must be a square matrix"
          << std::endl;
      status_ = InvalidInput;
      return;
    }
    etree_->BuildFromMatrix(A);
  }
  void Factorize() {

  }
  ~SimplicialCholesky() {
    delete etree_;
  }
private:
  using type = SimplicialCholesky<Derived, LDLT>;
  using Base = SparseCholeskyBase<type>;
  using Base::status_;
};

template <typename Derived, bool LDLT>
struct traits<SimplicialCholesky<Derived, LDLT>> {
  using MatDerived = Derived;
};

template <typename Derived>
using SimplicialLLT = SimplicialCholesky<Derived, false>;
template <typename Derived>
using SimplicialLDLT = SimplicialCholesky<Derived, true>;
} // namespace spmx

#endif // SPMX_SPMX_CHOLESKY_H