//
// Created by creeper on 23-2-23.
//

#ifndef SPMX_SPARSE_MATRIX_H
#define SPMX_SPARSE_MATRIX_H

#include "my-stl.h"
#include "sparse-storage.h"
#include "spmx-utils.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <spmx-options.h>
#include <spmx-types.h>
#include <spmx-vector.h>
#include <vector>

namespace spmx {
/**
 * Beside dynamic sparse matrix, specifically designed static sparse matrix is
 * also provided. When constructing static sparse matrix, the shape of the
 * matrix must be indicated, and the elements must be given in a Triplet array.
 * When it is created, the storage will be refined automatically for future use.
 * Such matrix is particularly useful for physically based animation, where the
 * matrix will not change after created. And the operations can be optimized
 * specifically.
 */
template <typename Derived, StorageType Storage> class SparseMatrixBase {


  Derived& operator+=(const Derived& other) {

    return static_cast<Derived&>(*this);
  }
  Derived& operator*=(const Derived& other) {

    return static_cast<Derived&>(*this);
  }
};

template <StorageType Storage, StorageMajoring MajorType>
class SparseMatrix
    : public SparseMatrixBase<SparseMatrix<Storage, MajorType>, Storage> {
public:
  SparseMatrix() = default;
  SparseMatrix(uint nRows, uint nCols, uint nnz)
      : n_rows_(nRows), n_cols_(nCols), nnz_(nnz), storage_(nnz) {
    if constexpr (MajorType == RowMajor)
      outer_idx_ = new uint[nRows + 1];
    else if constexpr (MajorType == ColumnMajor)
      outer_idx_ = new uint[nCols + 1];
  }

  template <StorageType AnyType, StorageMajoring AnyMajor>
  explicit SparseMatrix(const SparseMatrix<AnyType, AnyMajor> &spm)
      : n_rows_(spm.n_rows_), n_cols_(spm.n_cols_), nnz_(spm.nnz_) {
    if constexpr (MajorType == AnyMajor) {
      outer_idx_ = new uint[n_rows_ + 1];
      memcpy(n_rows_, spm.n_rows_, sizeof(Real) * spm.n_cols_);
      storage_ = spm.storage_;
    } else if constexpr (MajorType != AnyMajor) {

    }
  }

  template <StorageType AnyStorage, StorageMajoring AnyMajor>
  explicit SparseMatrix(SparseMatrix<AnyStorage, AnyMajor> &&spm)
      : n_rows_(spm.n_rows_), n_cols_(spm.n_cols_), nnz_(spm.nnz_) {
    if constexpr (MajorType == AnyMajor) {
      outer_idx_ = std::move(spm.outer_idx_);
      storage_ = spm.storage_;
    }
  }
  uint* OuterIndices() const {
    return outer_idx_;
  }
  uint* InnerIndices() const {
    return storage_.InnerIndices();
  }
  uint InnerIndex(uint i) const {
    return storage_.InnerIdx(i);
  }
  uint Data(uint i) const {
    return storage_.Data(i);
  }
  Real* Datas() const {
    return storage_.Datas();
  }
  SparseMatrix<Storage, MajorType ^ 1>
private:
  uint n_rows_ = 0, n_cols_ = 0, nnz_ = 0;
  uint *outer_idx_ = nullptr;
  SparseStorage storage_;
};

} // namespace spmx

#endif // SPMX_SPARSE_MATRIX_H
