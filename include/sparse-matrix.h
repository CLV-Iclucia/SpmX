//
// Created by creeper on 23-2-23.
//

#ifndef SPMX_SPARSE_MATRIX_H
#define SPMX_SPARSE_MATRIX_H

#include <cassert>
#include <cstring>
#include <iostream>
#include <sparse-matrix-base.h>
#include <sparse-storage.h>
#include <spmx-utils.h>
#include <utility>

namespace spmx {

template <uint nRows, uint nCols, StorageType Storage, StorageMajor Major>
class SparseMatrix;

/**
 * These vector classes are HIGHLY RECOMMENDED, specifications and optimizations
 * are provided for these vectors.
 * By default we use column vector. So it is recommended to mul the matrix on
 * the left
 */
template <StorageType Storage, uint Size = 0>
using Vector = SparseMatrix<Size, 1, Storage, ColMajor>;
template <StorageType Storage, uint Size = 0>
using ColVector = SparseMatrix<Size, 1, Storage, ColMajor>;
template <StorageType Storage, uint Size = 0>
using RowVector = SparseMatrix<1, Size, Storage, RowMajor>;

/**
 * General SparseMatrix class
 * @tparam nRows 0 for determination at runtime, otherwise determined at
 * compile time
 * @tparam nCols 0 for determination at runtime, otherwise determined at
 * compile time
 * @tparam Storage whether use sparse storage or dense storage for inner
 * indices.
 * @note for outer indices it must be a sparse storage
 * @tparam Major whether use row Majoring or column Majoring
 */
template <uint nRows, uint nCols, StorageType storageType, StorageMajor Major>
class SparseMatrix
    : public SparseMatrixBase<SparseMatrix<nRows, nCols, storageType, Major>> {
public:
  static_assert(
      Major != Symmetric || !nRows || !nCols || nRows == nCols,
      "Error: a matrices declared to be symmetric explicitly must be a square");
  using Base = SparseMatrixBase<SparseMatrix<nRows, nCols, storageType, Major>>;
  using Base::toTriplets;
  using Base::operator*;
  using Base::operator+;
  using Base::operator-;
  using Base::operator*=;
  using Base::operator+=;
  using Base::operator-=;
  static constexpr bool outer_idx_enabled =
      traits<SparseMatrix>::storage == Sparse &&
      !is_supported_vector<SparseMatrix>;
  SparseMatrix() = default;
  void SetShape(uint n_rows, uint n_cols) {
    assert((!nRows || nRows == n_rows) && (!nCols || nCols == n_cols));
    if constexpr (Major == Symmetric) {
      if constexpr (nRows)
        n_cols_ = nRows;
      else if constexpr (nCols)
        n_rows_ = nCols;
      else {
        assert(n_rows == n_cols);
        n_rows_ = n_rows;
        n_cols_ = n_cols;
      }
      return;
    }
    if constexpr (!nRows)
      n_rows_ = n_rows;
    if constexpr (!nCols)
      n_cols_ = n_cols;
  }
  void ReserveDenseStorage(uint dim) {
    if (storageType == Dense) {
      data_ = new Real[dim];
#ifdef MEMORY_TRACING
      MEMORY_LOG_ALLOC(SparseMatrix, dim);
#endif
      memset(data_, 0, sizeof(Real) * dim);
    }
  }

  // for building a sparse matrix, or a supported vector
  SparseMatrix(uint n_rows, uint n_cols) {
    SetShape(n_rows, n_cols);
    uint dim;
    if constexpr (Major == Symmetric) {
      assert(n_rows == n_cols);
      dim = n_rows;
    }
    if constexpr (Major == RowMajor)
      dim = n_cols;
    if constexpr (Major == ColMajor)
      dim = n_rows;
    if constexpr (is_supported_vector<SparseMatrix>) {
      ReserveDenseStorage(dim);
      return;
    }
    if constexpr (storageType == Sparse) {
      outer_idx_ = new uint[dim + 1];
#ifdef MEMORY_TRACING
      outer_idx_used = dim + 1;
      MEMORY_LOG_ALLOC(SparseMatrix, dim + 1);
#endif
    } else
      ReserveDenseStorage(n_rows * n_cols);
  }

  uint Dim(uint n_rows, uint n_cols) {
    if constexpr (Major == Symmetric) {
      assert(n_rows == n_cols);
      return n_rows;
    }
    if constexpr (Major == RowMajor)
      return n_rows;
    if constexpr (Major == ColMajor)
      return n_cols;
  }

  // only enabled for sparse storage
  SparseMatrix(uint n_rows, uint n_cols, uint nnz) : nnz_(nnz), storage_(nnz) {
    static_assert(storageType == Sparse);
    SetShape(n_rows, n_cols);
    uint dim = Dim(n_rows, n_cols);
    if constexpr (!is_supported_vector<SparseMatrix>) {
      outer_idx_ = new uint[dim + 1];
#ifdef MEMORY_TRACING
      outer_idx_used = dim + 1;
      MEMORY_LOG_ALLOC(SparseMatrix, dim + 1);
#endif
      memset(outer_idx_, 0, sizeof(uint) * (dim + 1));
    }
  }
  /**
   * Make sure that outer_idx_ are allocated properly
   * @tparam Iterator
   * @param begin
   * @param end
   */
  template <typename Iterator>
  void SetFromTriplets(Iterator begin, Iterator end, uint Options = 0) {
    if constexpr (storageType == Dense) {
      memset(data_, 0, sizeof(Real) * Dim());
      for (Iterator it = begin; it != end; it++)
        operator()(std::get<0>(*it), std::get<1>(*it)) += std::get<2>(*it);
    } else {
      if (!(Options & Ordered))
        std::sort(begin, end, TripletCmp<Major>);
      nnz_ = 0;
      uint outer_idx = 0;
      for (Iterator it = begin; it != end; it++)
        nnz_++;
      storage_.Reserve(nnz_);
      storage_.Reset(nnz_);
      nnz_ = 0;
      if (Options & NoRepeat) {
        for (Iterator it = begin; it != end; it++) {
          uint outer = (Major == RowMajor || Major == Symmetric)
                           ? std::get<0>(*it)
                           : std::get<1>(*it);
          uint inner = (Major == RowMajor || Major == Symmetric)
                           ? std::get<1>(*it)
                           : std::get<0>(*it);
          while (outer >= outer_idx) {
            outer_idx_[outer_idx] = nnz_;
            outer_idx++;
          }
          storage_.InnerIdx(nnz_) = inner;
          storage_.Data(nnz_) = std::get<2>(*it);
          nnz_++;
        }
      } else {
        uint cur_outer, cur_inner;
        Iterator it = begin;
        while (it != end) {
          uint outer = (cur_outer = (Major == RowMajor || Major == Symmetric)
                                       ? std::get<0>(*it)
                                       : std::get<1>(*it));
          uint inner = (cur_inner = (Major == RowMajor || Major == Symmetric)
                                       ? std::get<1>(*it)
                                       : std::get<0>(*it));
          while (outer >= outer_idx) {
            outer_idx_[outer_idx] = nnz_;
            outer_idx++;
          }
          storage_.InnerIdx(nnz_) = inner;
          while (outer == cur_outer && inner == cur_inner) {
            storage_.Data(nnz_) += std::get<2>(*it);
            ++it;
            if (it == end)
              break;
            outer = Major == RowMajor || Major == Symmetric ? std::get<0>(*it)
                                                            : std::get<1>(*it);
            inner = Major == RowMajor || Major == Symmetric ? std::get<1>(*it)
                                                            : std::get<0>(*it);
          }
          nnz_++;
        }
      }
      storage_.SetUsed(nnz_);
      for (uint i = outer_idx; i <= OuterDim(); i++)
        outer_idx_[i] = nnz_;
    }
  }

  template <bool SetOuter, typename InputIterator>
  void SetStorageByIterator(InputIterator &it) {
    uint outer_cnt = 0;
    nnz_ = 0;
    while (it()) {
      if constexpr (SetOuter) {
        while (outer_cnt <= it.Outer())
          outer_idx_[outer_cnt++] = nnz_;
      }
      storage_.InnerIdx(nnz_) = it.Inner();
      storage_.Data(nnz_) = it.value();
      nnz_++;
      ++it;
    }
    if constexpr (SetOuter) {
      while (outer_cnt <= OuterDim())
        outer_idx_[outer_cnt++] = nnz_;
    }
    storage_.SetUsed(nnz_);
  }

  template <typename OtherDerived> void CastToDense(const OtherDerived &spm) {
    static_assert(storageType == Dense,
                  "Error: CastToDense can only be called for dense storage");
    data_ = new Real[Dim()];
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(SparseMatrix, Dim());
#endif
    if constexpr (traits<OtherDerived>::storage == Sparse) {
      memset(data_, 0, sizeof(Real) * Dim());
      typename OtherDerived::NonZeroIterator it(spm);
      SetByIterator(it);
      return;
    }
    if constexpr (is_same_major<SparseMatrix, OtherDerived>) {
      for (uint i = 0; i < spm.OuterDim(); i++)
        for (uint j = 0; j < spm.InnerDim(); j++)
          AccessByMajor(i, j) = spm.AccessByMajor(i, j);
    } else {
      /**
       * TODO: optimize this!
       */
      for (uint i = 0; i < spm.OuterDim(); i++)
        for (uint j = 0; j < spm.InnerDim(); j++)
          AccessByMajor(j, i) = spm.AccessByMajor(i, j);
    }
  }
  /**
   * Specially designed for constructing a sparse matrix directly from
   * expressions
   * @tparam OtherDerived
   * @param spm
   */
  template <typename OtherDerived>
  explicit SparseMatrix(const SparseMatrixBase<OtherDerived> &other) {
    const OtherDerived &spm = other.derived();
    assert((!nRows || nRows == spm.Rows()) && (!nCols || nCols == spm.Cols()));
    SetShape(spm.Rows(), spm.Cols());
    if constexpr (storageType == Dense) { // dense mat/vec
      CastToDense(spm);
    } else if constexpr (is_supported_vector<SparseMatrix>) { // sparse vec
      if constexpr (traits<OtherDerived>::storage == Sparse) {
        typename OtherDerived::NonZeroIterator it(spm);
        storage_.Reserve(spm.NonZeroEst());
        SetStorageByIterator<false>(it);
      } else {
        for (int i = 0; i < spm.Dim(); i++) {
          if (!iszero(spm(i))) {
            storage_.InnerIdx(nnz_) = i;
            storage_.Data(nnz_++) = spm(i);
          }
        }
        storage_.SetUsed(nnz_);
      }
    } else {
      static_assert(traits<OtherDerived>::storage == Sparse,
                    "Error: current version do not support casting from dense "
                    "matrix to sparse matrix");
      if constexpr (is_same_major<SparseMatrix,
                                  OtherDerived>) { // sparse, same major
#ifdef MEMORY_TRACING
        outer_idx_used = spm.OuterDim() + 1;
        MEMORY_LOG_ALLOC(SparseMatrix, outer_idx_used);
#endif
        outer_idx_ = new uint[spm.OuterDim() + 1];
        typename OtherDerived::NonZeroIterator it(spm);
        storage_.Reserve(spm.NonZeroEst());
        SetByIterator(it);
      } else if constexpr (Major != traits<OtherDerived>::StorageMajor) {
        uint outer_dim = OuterDim(), inner_dim = InnerDim();
#ifdef MEMORY_TRACING
        outer_idx_used = outer_dim + 1;
        MEMORY_LOG_ALLOC(SparseMatrix, outer_idx_used);
#endif
        outer_idx_ = new uint[outer_dim + 1];
        Array<uint> bucket(outer_dim);
        storage_.Reserve(nnz_);
        memset(outer_idx_, 0, sizeof(uint) * (outer_dim + 1));
        bucket.Fill(0);
        for (typename OtherDerived::NonZeroIterator it(spm); it(); ++it)
          OuterIdx(it.InnerIdx())++;
        for (uint i = 1; i <= inner_dim; i++)
          OuterIdx(i) += OuterIdx(i - 1);
        for (uint i = 0; i < outer_dim; i++) {
          uint cnt = 0;
          for (typename OtherDerived::InnerIterator it(spm, i); it();
               ++it, cnt++) {
            uint idx =
                OuterIdx(it.Inner()) + bucket(InnerIdx(cnt + OuterIdx(i)));
            InnerIdx(idx) = i;
            Data(idx) = it.value();
            bucket[it.Inner()]++;
          }
        }
      }
    }
  }

  SparseMatrix(const SparseMatrix &spm)
      : n_rows_(spm.Rows()), n_cols_(spm.Cols()) {
    if constexpr (outer_idx_enabled) {
#ifdef MEMORY_TRACING
      MEMORY_LOG_ALLOC(SparseMatrix, spm.OuterDim() + 1);
      outer_idx_used = spm.OuterDim() + 1;
#endif
      outer_idx_ = new uint[spm.OuterDim() + 1];
      memcpy(outer_idx_, spm.outer_idx_, sizeof(uint) * (spm.OuterDim() + 1));
    }
    if constexpr (storageType == Sparse) {
      nnz_ = spm.NonZeroEst();
      storage_ = SparseStorage(nnz_);
    } else {
      data_ = new Real[spm.Dim()];
#ifdef MEMORY_TRACING
      MEMORY_LOG_ALLOC(SparseMatrix, spm.Dim());
#endif
      memcpy(data_, spm.data_, sizeof(Real) * spm.Dim());
    }
  }

  uint Rows() const {
    if constexpr (nRows)
      return nRows;
    else {
      if constexpr (Major == Symmetric && nCols)
        return nCols;
      return n_rows_;
    }
  }
  uint Cols() const {
    if constexpr (nCols)
      return nCols;
    else {
      if constexpr (Major == Symmetric && nRows)
        return nRows;
      return n_cols_;
    }
  }

  void ReallocStorage(uint new_outer_dim, uint new_dim, uint nnz) {
    if constexpr (outer_idx_enabled) {
      if (new_outer_dim > OuterDim()) {
        delete[] outer_idx_;
#ifdef MEMORY_TRACING
        MEMORY_LOG_DELETE(SparseMatrix, outer_idx_used);
        outer_idx_used = new_outer_dim + 1;
        MEMORY_LOG_ALLOC(SparseMatrix, outer_idx_used);
#endif
        outer_idx_ = new uint[new_outer_dim + 1];
      }
    }
    if constexpr (storageType == Sparse) {
      storage_.Reserve(nnz);
    } else {
      if (new_dim > Dim()) {
        delete[] data_;
#ifdef MEMORY_TRACING
        MEMORY_LOG_DELETE(SparseMatrix, Dim());
        MEMORY_LOG_ALLOC(SparseMatrix, new_dim);
#endif
        data_ = new Real[new_dim];
      }
    }
  }

  template <typename OtherDerived>
  SparseMatrix &operator=(const OtherDerived &other) {
    static_assert(is_same_shape_v<SparseMatrix, OtherDerived>);
    ReallocStorage(other.OuterDim(), other.Dim(), other.NonZeroEst());
    SetShape(other.Rows(), other.Cols());
    if constexpr (traits<OtherDerived>::storage == Sparse) {
      nnz_ = other.NonZeroEst();
      typename OtherDerived::NonZeroIterator it(other);
      SetByIterator(it);
    } else {
      for (uint i = 0; i < Dim(); i++)
        operator()(i) = other.operator()(i);
    }
    return *this;
  }

  SparseMatrix &operator=(const SparseMatrix &other) {
    if (&other == this)
      return *this;
    ReallocStorage(other.OuterDim(), other.Dim(), other.NonZeroEst());
    if constexpr (traits<SparseMatrix>::storage == Sparse) {
      if constexpr (!is_supported_vector<SparseMatrix>) {
        memcpy(outer_idx_, other.outer_idx_,
               sizeof(uint) * (other.OuterDim() + 1));
      }
      nnz_ = other.NonZeroEst();
      storage_ = other.storage_;
      storage_.SetUsed(nnz_);
    } else {
      memcpy(data_, other.data_, sizeof(Real) * other.Dim());
    }
    SetShape(other.Rows(), other.Cols());
    return *this;
  }

  SparseMatrix(SparseMatrix &&other) noexcept {
    if constexpr (outer_idx_enabled) {
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseMatrix, outer_idx_used);
#endif
      delete[] outer_idx_;
      outer_idx_ = other.outer_idx_;
      other.outer_idx_ = nullptr;
    }
    if constexpr (traits<SparseMatrix>::storage == Sparse) {
      nnz_ = other.nnz_;
      storage_ = std::move(other.storage_);
    } else {
      delete[] data_;
      data_ = other.data_;
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseMatrix, Dim());
#endif
      other.data_ = nullptr;
    }
    SetShape(other.Rows(), other.Cols());
  }

  void Prune() { nnz_ = outer_idx_[OuterDim()]; }
  SparseMatrix &operator=(SparseMatrix &&other) noexcept {
    if (&other == this)
      return *this;
    SetShape(other.Rows(), other.Cols());
    if constexpr (outer_idx_enabled) {
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseMatrix, outer_idx_used);
#endif
      delete[] outer_idx_;
      outer_idx_ = other.outer_idx_;
      other.outer_idx_ = nullptr;
    }
    if constexpr (storageType == Sparse) {
      nnz_ = other.nnz_;
      storage_ = std::move(other.storage_);
    } else {
      delete[] data_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  // Only allowed for dense vector. Constructor for an n-dim dense vector
  explicit SparseMatrix(uint n) {
    static_assert(storageType == Dense && is_supported_vector<SparseMatrix>);
    if constexpr (is_supported_vector_of_major_v<SparseMatrix, RowMajor>)
      n_cols_ = n;
    else
      n_rows_ = n;
    data_ = new Real[n];
  }

  /**
   * @param m
   * @param n
   */
  void Resize(uint m, uint n) {
    if constexpr (storageType == Sparse) {
      uint outer_dim = OuterDim();
      uint new_outer_dim = Dim(m, n);
      SetShape(m, n);
      if (outer_dim < new_outer_dim) {
        uint *new_outer_idx = new uint[new_outer_dim + 1];
        if (outer_idx_ != nullptr) {
          memcpy(new_outer_idx, outer_idx_, sizeof(uint) * (outer_dim + 1));
#ifdef MEMORY_TRACING
          MEMORY_LOG_DELETE(SparseMatrix, outer_idx_used);
#endif
          delete[] outer_idx_;
        }
#ifdef MEMORY_TRACING
        MEMORY_LOG_ALLOC(SparseMatrix, new_outer_dim + 1);
        outer_idx_used = new_outer_dim + 1;
#endif
        outer_idx_ = new_outer_idx;
      }
    } else {
      if (m * n > Dim()) { // if we need more space then we need to realloc
        uint dim = Dim();
        SetShape(m, n);
        Real *new_data = new Real[m * n];
        memcpy(new_data, data_, sizeof(Real) * dim);
        delete[] data_;
        data_ = new_data;
      }
    }
  }

  Real operator[](uint i) const {
    static_assert(storageType == Dense);
    return data_[i];
  }

  Real &operator[](uint i) {
    static_assert(storageType == Dense);
    return data_[i];
  }

  using TransposedMatrix =
      SparseMatrix<nCols, nRows, storageType, transpose_v<Major>>;

  SparseMatrix Eval() const { return SparseMatrix(*this); }

  /**
   * The following methods are only enabled for dense storage
   * @param i
   * @param j
   * @return
   */
  Real AccessByMajor(uint i, uint j) const { return data_[i * OuterDim() + j]; }
  Real &AccessByMajor(uint i, uint j) { return data_[i * OuterDim() + j]; }
  /**
   * Set the sparse matrix based on the input iterator
   * it runs the iterator and set the corresponding
   * for now, the iterator and the matrix must be in the same storage major
   * TODO: Add support for different orderings
   * @tparam InputIterator
   * @param it
   */
  template <typename InputIterator>
  void SetByIterator(InputIterator &it, uint options = 0) {
    if constexpr (storageType == Sparse) {
      SetStorageByIterator<true>(it);
    } else if (storageType == Dense) {
      while (it()) {
        AccessByMajor(it.Outer(), it.Inner()) = it.value();
        ++it;
      }
    }
  }

  friend TransposedMatrix;
  TransposedMatrix Transposed() const {
    TransposedMatrix ret(n_cols_, n_rows_);
    if constexpr (storageType == Sparse) {
      ret.storage_ = storage_;
      memcpy(ret.outer_idx_, outer_idx_, sizeof(uint) * (nnz_ + 1));
    } else {
      memcpy(ret.data_, data_, sizeof(uint) * Dim());
    }
    return ret;
  }

  template <StorageType VecStorage> Vector<VecStorage> Diag() const {
    Vector<VecStorage> diag(n_rows_);
    uint cnt = 0;
    for (uint i = 0; i < n_rows_; i++) {
      int idx = IndexAt(i, i);
      if constexpr (VecStorage == Dense) {
        diag(i) = idx < 0 ? 0 : Storage().Data(static_cast<uint>(i));
      } else {
        if (idx < 0)
          continue;
        diag.Idx(cnt) = InnerIdx(idx);
        diag.Data(cnt) = Data(idx);
        cnt++;
      }
    }
    return diag;
  }
  Real operator()(uint i, uint j) const {
    if constexpr (storageType == Sparse) {
      int idx = IndexAt(i, j);
      return idx < 0 ? 0 : Storage().Data(static_cast<uint>(i));
    } else {
      if constexpr (Major == RowMajor || Major == Symmetric)
        return data_[i * Rows() + j];
      else
        return data_[j * Cols() + i];
    }
  }

  Real &operator()(uint i, uint j) {
    if constexpr (storageType == Sparse) {
      int idx = IndexAt(i, j);
      if (idx < 0) {
        std::cerr << "Warning: reference a zero element is not allowed."
                  << std::endl;
        exit(-1);
      }
      return Data(static_cast<uint>(idx));
    } else {
      if constexpr (Major == RowMajor || Major == Symmetric)
        return data_[i * Rows() + j];
      else
        return data_[j * Cols() + i];
    }
  }
  Real operator()(uint i) const {
    static_assert(traits<SparseMatrix>::storage == Dense);
    return data_[i];
  }

  Real &operator()(uint i) {
    static_assert(traits<SparseMatrix>::storage == Dense);
    return data_[i];
  }
  uint OuterDim() const {
    if constexpr (Major == RowMajor || Major == Symmetric)
      return Rows();
    else
      return Cols();
  }
  uint InnerDim() const {
    if constexpr (Major == RowMajor || Major == Symmetric)
      return Cols();
    else
      return Rows();
  }
  uint NonZeroEst() const {
    if constexpr (nRows && nCols)
      return nRows * nCols;
    else
      return nnz_;
  }
  uint Dim() const { return Rows() * Cols(); }
  const SparseStorage &Storage() const { return storage_; }
  SparseStorage& Storage() { return storage_; }
  uint *OuterIndices() const { return outer_idx_; }
  uint *InnerIndices() const { return storage_.InnerIndices(); }
  uint OuterIdx(uint i) const {
    if constexpr (is_supported_vector<SparseMatrix>) {
      if (i == 0)
        return 0;
      else if (i == 1)
        return nnz_;
      assert(false);
    }
    return outer_idx_[i];
  }
  uint &OuterIdx(uint i) { return outer_idx_[i]; }
  uint InnerIdx(uint i) const { return storage_.InnerIdx(i); }
  uint &InnerIdx(uint i) { return storage_.InnerIdx(i); }
  Real Data(uint i) const {
    if constexpr (storageType == Sparse)
      return storage_.Data(i);
    else return data_[i];
  }
  Real &Data(uint i) {
    if constexpr (storageType == Sparse)
      return storage_.Data(i);
    else return data_[i];
  }
  Real *Datas() const {
    if constexpr (storageType == Sparse)
      return storage_.Datas();
    else return data_;
  }
  int IndexAt(uint i, uint j) const {
    uint idx = storage_.SearchIndex(OuterIdx(), j);
    return idx == OuterIdx() ? -1 : static_cast<int>(idx);
  }

  class InnerIterator {
  public:
    InnerIterator(const SparseMatrix &mat, uint outer_idx)
        : cur_(mat.OuterIdx(outer_idx)), end_(mat.OuterIdx(outer_idx + 1)),
          storage_(mat.Storage()) {}
    uint Inner() const { return storage_.InnerIdx(cur_); }
    Real value() const { return storage_.Data(cur_); }
    bool operator()() const { return cur_ < end_; }
    [[maybe_unused]] InnerIterator &operator++() {
      cur_++;
      return *this;
    }

  private:
    uint cur_;
    uint end_;
    const SparseStorage &storage_;
  };

  class NonZeroIterator {
  public:
    explicit NonZeroIterator(const SparseMatrix &spm) {
      static_assert(is_supported_vector<SparseMatrix> || storageType == Sparse,
                    "Error: accessing dense matrix by is not supported yet");
      if constexpr (traits<SparseMatrix>::storage == Sparse) {
        inner_ = spm.storage_.InnerIndices();
        data_ = spm.storage_.Datas();
        nnz_ = spm.nnz_;
      } else {
        data_ = spm.data_;
        nnz_ = spm.Dim();
      }
      if constexpr (!is_supported_vector<SparseMatrix>) {
        outer_idx_ = spm.OuterIndices();
        while (outer_idx_[outer_ptr_ + 1] == 0)
          outer_ptr_++;
      }
    }
    uint Row() const {
      if constexpr (Major == RowMajor || Major == Symmetric)
        return Outer();
      else
        return Inner();
    }
    uint Col() const {
      if constexpr (Major == RowMajor || Major == Symmetric)
        return Inner();
      else
        return Outer();
    }
    uint Outer() const { return outer_ptr_; }
    uint Inner() const {
      if constexpr (storageType == Sparse) {
        return inner_[inner_ptr_];
      } else if constexpr (is_supported_vector<SparseMatrix>) {
        return inner_ptr_;
      }
    }
    Real value() const { return data_[inner_ptr_]; }
    Real &value() { return data_[inner_ptr_]; }
    bool operator()() const { return inner_ptr_ < nnz_; }
    [[maybe_unused]] NonZeroIterator &operator++() {
      inner_ptr_++;
      if constexpr (!is_supported_vector<SparseMatrix>) {
        while (inner_ptr_ >= outer_idx_[outer_ptr_ + 1]) {
          outer_ptr_++;
        }
      }
      return *this;
    }

  private:
    uint inner_ptr_ = 0, outer_ptr_ = 0;
    uint nnz_ = 0;
    uint *inner_ = nullptr;
    uint *outer_idx_ = nullptr;
    Real *data_ = nullptr;
  };

  template <typename Func> void ForEachNonZeros(Func &&f) {
    for (NonZeroIterator it(*this); it(); ++it)
      it.value() = f(it.value());
  }

  bool IsSquare() const { return Rows() == Cols(); }

  friend std::ostream &operator<<(std::ostream &o, const SparseMatrix &spm) {
    NonZeroIterator it(spm);
    while (it()) {
      o << it.Row() << ", " << it.Col() << ", " << it.value() << std::endl;
      ++it;
    }
    return o;
  }

  ~SparseMatrix() {
    if constexpr (storageType == Sparse) {
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseMatrix, outer_idx_used);
#endif
      delete[] outer_idx_;
    } else {
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseMatrix, Dim());
#endif
      delete[] data_;
    }
  }

protected:
#ifdef MEMORY_TRACING
  uint outer_idx_used = 0;
#endif
  finalized_when_t<nRows != 0, uint> n_rows_ = nRows;
  finalized_when_t<nCols != 0, uint> n_cols_ = nCols;
  finalized_when_t<is_supported_vector<SparseMatrix> || storageType == Dense,
                   uint *>
      outer_idx_ = nullptr;
  finalized_when_t<storageType == Sparse, Real *> data_ = nullptr;
  finalized_when_t<storageType == Dense, uint> nnz_ = 0;
  finalized_when_t<storageType == Dense, SparseStorage> storage_;
};

using SparseMatrixXd = SparseMatrix<0, 0, Sparse, RowMajor>;
template <uint N> using SparseSquare = SparseMatrix<N, N, Sparse, RowMajor>;
template <uint M, uint N>
using SparseMat = SparseMatrix<M, N, Sparse, RowMajor>;
using SymSparseMatrixXd = SparseMatrix<0, 0, Sparse, Symmetric>;
template <uint N> using SymSparseMatrix = SparseMatrix<N, N, Sparse, Symmetric>;

template <uint nRows, uint nCols, StorageMajor Major>
class TripletSparseMatrix
    : SparseMatrixBase<TripletSparseMatrix<nRows, nCols, Major>> {
public:
  TripletSparseMatrix(uint n_rows, uint n_cols, uint nnz)
      : n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz) {
    outer_idx_ = new uint[nnz];
    inner_idx_ = new uint[nnz];
    data_ = new Real[nnz];
  }
  template <typename Iterator>
  void SetFromTriplets(Iterator begin, Iterator end) {
    // TODO: Implement SetFromTriplets
  }
  uint OuterIdx(uint i) const { return outer_idx_[i]; }
  uint InnerIdx(uint i) const { return inner_idx_[i]; }
  uint Data(uint i) const { return data_[i]; }

  class NonZeroIterator;
  ~TripletSparseMatrix() {
    delete[] outer_idx_;
    delete[] inner_idx_;
    delete[] data_;
  }

private:
  uint nnz_ = 0;
  finalized_when_t<nRows, uint> n_rows_ = nRows;
  finalized_when_t<nCols, uint> n_cols_ = nCols;
  uint *outer_idx_ = nullptr;
  uint *inner_idx_ = nullptr;
  Real *data_ = nullptr;
};

template <uint nRows_, uint nCols_, StorageType Storage_, StorageMajor Major_>
struct traits<SparseMatrix<nRows_, nCols_, Storage_, Major_>> {
  using EvalType = SparseMatrix<nRows_, nCols_, Storage_, Major_>;
  static constexpr uint nRows = nRows_;
  static constexpr uint nCols = nCols_;
  static constexpr StorageType storage = Storage_;
  static constexpr StorageMajor major = Major_;
};

template <uint nRows_, uint nCols_, StorageMajor Major_>
struct traits<TripletSparseMatrix<nRows_, nCols_, Major_>> {
  using EvalType = SparseMatrix<nRows_, nCols_, Sparse, Major_>;
  static constexpr uint nRows = nRows_;
  static constexpr uint nCols = nCols_;
  static constexpr StorageType storage = Sparse;
  static constexpr StorageMajor major = Major_;
};

template <uint nRows, uint nCols, StorageMajor Major>
class TripletSparseMatrix<nRows, nCols, Major>::NonZeroIterator {
public:
  using SparseMatrix = TripletSparseMatrix<nRows, nCols, Major>;
  NonZeroIterator(const SparseMatrix &mat, uint begin, uint end)
      : cur_(begin), end_(end), outer_idx_(mat.outer_idx_),
        inner_idx_(mat.inner_idx_), data_(mat.data_) {}
  uint Outer() const { return outer_idx_[cur_]; }
  uint Inner() const { return inner_idx_[cur_]; }
  Real Data() const { return data_[cur_]; }
  bool operator()() const { return cur_ < end_; }
  NonZeroIterator &operator++() {
    cur_++;
    return *this;
  }

private:
  uint cur_;
  uint end_;
  uint *outer_idx_;
  uint *inner_idx_;
  Real *data_;
};
} // namespace spmx
#endif // SPMX_SPARSE_MATRIX_H
