//
// Created by creeper on 23-5-16.
//

#ifndef SPMX_SPMX_CHOLESKY_H
#define SPMX_SPMX_CHOLESKY_H

#include "my-stl.h"
#include <linear-solvers.h>
#include <sparse-matrix.h>

namespace spmx {

template <typename Derived>
class SparseCholeskyBase : public FactorizeSolver<SparseCholeskyBase<Derived>> {
public:
  using MatType = typename traits<Derived>::MatType;
  const Derived &derived() const { return *static_cast<Derived *>(this); }
  Derived &derived() { return *static_cast<Derived *>(this); }
  void IAnalyse(const MatType &A) { derived().IAnalyse(A); }
  void IFactorize() { derived().Factorize(); }
  void ICompute(const MatType &A) {
    derived().Analyse(A);
    if (status_ != Success)
      return;
    derived().Factorize(A);
    if (status_ != Success)
      return;
  }
  template <StorageType VecStorage>
  Vector<Dense> SolveImpl(const Vector<VecStorage> &b) {}

protected:
  using FactorizeSolver<SparseCholeskyBase<Derived>>::status_;
};

/**
 * This has a defect - for a sparse matrix, there is no difference in CSR and
 * CSC But we treat them as two different types
 * @tparam MatType to use the SimplicialCholesky solver, the matrix type must
 * implement an InnerIterator to iterate a inner dim
 * @tparam LDLT can be true of false, indicating whether to do a LDLT
 * factorization
 */
template <typename MatType, bool LDLT>
class SimplicialCholesky
    : SparseCholeskyBase<SimplicialCholesky<MatType, LDLT>> {
  using SolverType = SimplicialCholesky<MatType, LDLT>;
  static constexpr uint Size = traits<SolverType>::Size;

private:
  uint *etree_fa_ = nullptr; // for the largest index, its father on elimination
                             // tree is undefined, and shouldn't be accessed
  void BuildEtreeFromMat(const MatType &A) {
    Array<int> anc(A.Rows());
    anc.Fill(-1);
    for (uint i = 1; i < A.Rows(); i++) {
      for (uint j = A.OuterIndex(i);
           j < A.OuterIndex(i + 1) && A.InnerIdx(j) < i; j++) {
        uint x = A.InnerIdx(j);
        while (etree_fa_[x] >= 0) {
          uint t = anc[x];
          anc[x] = static_cast<int>(i);
          x = t;
        }
        etree_fa_[x] = anc[x] = static_cast<int>(i);
      }
    }
  }
  void ReallocEtree(uint size) {
    static_assert(!Size, "Error: Solvers for matrices of a fixed size cannot "
                         "reallocate etree array");
#ifdef MEMORY_TRACING
    MEMORY_LOG_REALLOC(SimplicialCholesky, etree_fa_size, size);
#endif
    delete[] etree_fa_;
    etree_fa_ = new uint[size];
  }

public:
  SimplicialCholesky() {
    if constexpr (Size) {
#ifdef MEMORY_TRACING
      etree_fa_size = Size;
      MEMORY_LOG_ALLOC(etree_fa_, Size);
#endif
      etree_fa_ = new uint[Size];
    }
  }
  void AnalyseImpl(const SparseMatrixBase<MatType> &A) {
    if (!A.IsSquare()) {
      std::cerr
          << "Error: Input matrix for Cholesky solver must be a square matrix"
          << std::endl;
      status_ = InvalidInput;
      return;
    }
    BuildEtreeFromMat(A);
  }
  /**
   * The analyzing pass can be wrapped in a solver, but it makes sense to
   * if the user knows that many matrices have the same sparse pattern, they can
   * just make one call to Analyze
   * @param A
   */
  void Analyze(const MatType &A) { BuildElimFromMatrix(A); }
  void Factorize() {}
  ~SimplicialCholesky() {
#ifdef MEMORY_TRACING
    if (etree_fa_ == nullptr)
      MEMORY_LOG_DELETE(SimplicialCholesky, "nullptr");
    else
      MEMORY_LOG_DELETE(SimplicialCholesky, etree_fa_size);
#endif
    delete etree_fa_;
  }

private:
#ifdef MEMORY_TRACING
  uint etree_fa_size = 0;
#endif
  using type = SimplicialCholesky<MatType, LDLT>;
  using Base = SparseCholeskyBase<type>;
  using Base::status_;
};

/**
 * For a Solver template class, traits<Solver> extracts the type and size of the
 * matrix of the Solver
 * @tparam Derived
 * @tparam LDLT
 */
template <typename MatType_, bool LDLT>
struct traits<SimplicialCholesky<MatType_, LDLT>> {
  static_assert(traits<MatType_>::nRows == traits<MatType_>::nCols);
  using TraitsMat = traits<MatType_>;
  static constexpr uint Size = TraitsMat::nRows | TraitsMat::nCols;
  using MatType = MatType_;
};

template <typename Derived>
using SimplicialLLT = SimplicialCholesky<Derived, false>;
template <typename Derived>
using SimplicialLDLT = SimplicialCholesky<Derived, true>;
} // namespace spmx

#endif // SPMX_SPMX_CHOLESKY_H