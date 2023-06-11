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
  static_assert(traits<MatType>::major == Symmetric &&
                    traits<MatType>::storage == Sparse,
                "Error: to use the sparse cholesky factorization solver, the "
                "matrix has to be declared explicitly as sparse and symmetric");
  const Derived &derived() const { return *static_cast<Derived *>(this); }
  Derived &derived() { return *static_cast<Derived *>(this); }
  void Analyse(const MatType &A) { derived().Analyse(A); }
  void Factorize(const MatType &A) { derived().Factorize(A); }
  void Compute(const MatType &A) {
    derived().Analyse(A);
    if (status_ != Success)
      return;
    derived().Factorize(A);
    if (status_ != Success)
      return;
  }
  template <StorageType VecStorage>
  Vector<Dense> Solve(const Vector<VecStorage> &b) {}

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
    : public SparseCholeskyBase<SimplicialCholesky<MatType, LDLT>> {
  using SolverType = SimplicialCholesky<MatType, LDLT>;
  static constexpr uint Size = traits<SolverType>::Size;

protected:
  void BuildEtreeFromMat(const MatType &A) {
    Array<int> anc(A.Rows());
    anc.Fill(-1);
    for (uint i = 1; i < A.Rows(); i++) {
      nnz_cnt_[i] = 1;
      for (typename MatType::InnerIterator it(A, i); it() && it.Inner() < i;
           ++it) {
        uint x = it.Inner();
        while (etree_fa_[x] >= 0) {
          uint t = anc[x];
          anc[x] = static_cast<int>(i);
          nnz_cnt_[x]++;
          x = t;
        }
        etree_fa_[x] = anc[x] = static_cast<int>(i);
      }
    }
    outer_idx_[0] = 0;
    for (uint i = 0; i < size_; i++) {
      outer_idx_[i + 1] = outer_idx_[i] + nnz_cnt_[i];
    }
    storage_.Reserve(outer_idx_[size_]);
    memset(nnz_cnt_, 0, sizeof(uint) * size_);
  }
  void Reserve(uint size) {
    static_assert(!Size, "Error: Solvers for matrices of a fixed size cannot "
                         "reallocate etree array");
#ifdef MEMORY_TRACING
    MEMORY_LOG_REALLOC(SimplicialCholesky, allocated_size, size);
    allocated_size = size;
#endif
    if (size > size_) {
      delete[] etree_fa_;
      etree_fa_ = new uint[size];
      delete[] outer_idx_;
      outer_idx_ = new uint[size + 1];
      if constexpr (LDLT) {
        delete[] diag_;
        diag_ = new Real[size];
      }
    nnz_cnt_ = new uint[size];
    }
  }
  void InitSolver(const MatType &A) {
    if (!A.IsSquare()) {
      std::cerr << "Error: solvers can only apply on square matrices"
                << std::endl;
      status_ = InvalidInput;
      return;
    }
    Reserve(A.Rows());
    size_ = A.Rows();
  }

public:
  SimplicialCholesky() {
    if constexpr (Size) {
#ifdef MEMORY_TRACING
      allocated_size = Size;
      MEMORY_LOG_ALLOC(etree_fa_, Size);
#endif
      etree_fa_ = new uint[Size];
      if constexpr (LDLT)
        diag_ = new uint[Size];
      outer_idx_ = new uint[Size + 1];
      nnz_cnt_ = new uint[Size];
    }
  }
  /**
   * The analysing pass can be wrapped in a solver, but it makes sense to make
   * it public.
   * if the user knows that many matrices have the same sparse pattern, they can
   * just make one call to Analyze
   * @param A
   */
  void Analyse(const MatType &A) {
    InitSolver(A);
    if (status_ == InvalidInput)
      return;
    BuildEtreeFromMat(A);
  }
  /**
   * Since A is symmetric, we can safely accessing it in any major.
   * @param A
   */
  void Factorize(const MatType &A) {
    Array<Real> sum(size_);
    Array<uint> mask(size_);
    Array<uint> row_sparsity_pattern(size_);
    mask.Fill(false);
    status_ = Success;
    for (uint i = 0; i < size_; i++) {
      uint cur_cnt = 0;
      for (typename MatType::InnerIterator it(A, i); it(); ++it) {
        uint k = it.Inner();
        if (k > i)
          break;
        sum(k) += it.value();
        uint j = k;
        while (j < i) {
          if (mask[j] != i) {
            mask[j] = i;
            row_sparsity_pattern[cur_cnt++] = j;
          }
          j = etree_fa_[j];
        }
      }
      Real diag_i = 0;
      for (uint j = 0; j < cur_cnt; j++) {
        uint k = row_sparsity_pattern[j];
        Real diag_k;
        if constexpr (LDLT)
          diag_k = diag_[k];
        else
          diag_k = storage_.Data(outer_idx_[k]);
        if (iszero(diag_k)) {
          status_ = NumericalError;
          return;
        }
        Real l_ik = sum(k) / diag_k;
        sum(k) = 0;
        uint p = outer_idx_[k] + (LDLT ? 0 : 1);
        for (; p < outer_idx_[k + 1]; p++)
          sum(storage_.InnerIdx(p)) -= l_ik * storage_.Data(p);
        storage_.InnerIdx(p) = i;
        storage_.Data(p) = l_ik;
        nnz_cnt_[k]++;
      }
      if (diag_i <= 0) {
        status_ = NumericalError;
        break;
      }
      if constexpr (LDLT) {
        diag_[i] = diag_i;
      } else {
        storage_.InnerIdx(outer_idx_[i]) = i;
        storage_.Data(outer_idx_[i]) = std::sqrt(diag_i);
        nnz_cnt_[i]++;
      }
    }
  }
  Vector<Dense, Size> Solve(const Vector<Dense, Size> &b) {
    Vector<Dense, Size> x(b);
    for (int i = 0; i < size_; i++) {
      if constexpr (LDLT)
        x(i) /= diag_(i);
      else
        x(i) /= storage_.Data(outer_idx_[i]);
      Real xi = x(i);
      for (int ptr = outer_idx_[i] + (LDLT ? 0 : 1); ptr < outer_idx_[i + 1];
           ptr++) {
        x(storage_.InnerIdx(ptr)) -= storage_.Data(ptr) * xi;
      }
    }
    for (int i = size_ - 1; i >= 0; i--) {
      for (int ptr = outer_idx_[i] + (LDLT ? 0 : 1); ptr < outer_idx_[i + 1];
           ptr++) {
        x(i) -= storage_.Data(ptr) * x(storage_.InnerIdx(ptr));
      }
      if constexpr (LDLT)
        x(i) /= diag_(i);
      else
        x(i) /= storage_.Data(outer_idx_[i]);
    }
    return x;
  }
  ~SimplicialCholesky() {
#ifdef MEMORY_TRACING
    if (etree_fa_ == nullptr)
      MEMORY_LOG_DELETE(SimplicialCholesky, "nullptr");
    else
      MEMORY_LOG_DELETE(SimplicialCholesky, allocated_size);
#endif
    delete etree_fa_;
    delete outer_idx_;
    if constexpr (LDLT)
      delete diag_;
  }

protected:
#ifdef MEMORY_TRACING
  uint allocated_size = 0;
#endif
  finalized_when_t<Size != 0, uint> size_ = Size;
  finalized_when_t<LDLT, Real *> diag_ = nullptr;
  uint *nnz_cnt_ = nullptr;
  uint *outer_idx_ = nullptr;
  uint *etree_fa_ = nullptr; // for the largest index, its father on elimination
                             // tree is undefined, and shouldn't be accessed
  SparseStorage storage_;
  using Base = SparseCholeskyBase<SimplicialCholesky>;
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