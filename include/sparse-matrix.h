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
#include <utility>

namespace spmx {

template <uint nRows, uint nCols, StorageType Storage, StorageMajor Major>
class SparseMatrix;

template <typename Lhs, typename Rhs> class AddExpr;

template <StorageType Storage>
using Vector = SparseMatrix<0, 1, Storage, ColMajor>;
template <StorageType Storage>
using ColVector = SparseMatrix<0, 1, Storage, ColMajor>;
template <StorageType Storage>
using RowVector = SparseMatrix<1, 0, Storage, RowMajor>;

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
  using Base = SparseMatrixBase<SparseMatrix<nRows, nCols, storageType, Major>>;
  using Base::toTriplets;
  using Base::operator+;
  using Base::operator-;
  SparseMatrix() = default;
  SparseMatrix(uint n_rows, uint n_cols, uint nnz) {
    if constexpr (!nRows)
      n_rows_ = n_rows;
    if constexpr (!nCols)
      n_cols_ = n_cols;
    if (storageType == Sparse) {
      if constexpr (Major == RowMajor)
        outer_idx_ = new uint[n_rows + 1];
      else if constexpr (Major == ColMajor)
        outer_idx_ = new uint[n_cols + 1];
      nnz_ = nnz;
    } else {
      data_ = new Real[n_rows_ * n_cols_];
    }
  }
  /**
   * By default, the space should be allocated
   * @tparam Iterator
   * @param begin
   * @param end
   */
  template <typename Iterator>
  void SetFromTriplets(Iterator begin, Iterator end) {

  }
  template <typename OtherDerived>
  explicit SparseMatrix(const SparseMatrixBase<OtherDerived> &spm) {
    if constexpr (Major == traits<OtherDerived>::StorageMajor) {
      outer_idx_ = new uint[n_rows_ + 1];
      memcpy(n_rows_, spm.n_rows_, sizeof(Real) * spm.n_cols_);
      storage_ = spm.storage_;
    } else if constexpr (Major != traits<OtherDerived>::StorageMajor) {
      uint outer_dim = OuterDim(), inner_dim = InnerDim();
      outer_idx_ = new uint[outer_dim + 1];
      uint *bucket = new uint[outer_dim];
      storage_ = SparseStorage(nnz_);
      memset(outer_idx_, 0, sizeof(uint) * (outer_dim + 1));
      memset(bucket, 0, sizeof(uint) * outer_dim);
      for (uint i = 0; i < nnz_; i++)
        OuterIdx(i)++;
      for (uint i = 1; i <= inner_dim; i++)
        OuterIdx(i) += OuterIdx(i - 1);
      for (uint i = 0; i < outer_dim; i++) {
        for (uint j = OuterIdx(i); j < OuterIdx(i + 1); j++) {
          uint idx = OuterIdx();
          Storage().InnerIdx(idx) = i;
          Storage().Data(idx) = spm.Data(j);
          bucket[spm.InnerIdx(j)]++;
        }
      }
      delete[] bucket;
    }
  }
  uint Rows() const {
    if constexpr (!nRows)
      return n_rows_;
    else
      return nRows;
  }
  uint Cols() const {
    if constexpr (!nCols)
      return n_cols_;
    else
      return nCols;
  }
  template <typename OtherDerived>
  SparseMatrix &operator=(const OtherDerived &other) {
    n_rows_ = other.Rows();
    n_cols_ = other.Cols();
    nnz_ = other.NonZeroEst();
    if (other.OuterDim() > OuterDim()) {
      delete[] outer_idx_;
      outer_idx_ = new uint[other.OuterDim()];
    }
    typename OtherDerived::NonZeroIterator it(other);
    SetByIterator(it);
    return *this;
  }

  SparseMatrix &operator=(const SparseMatrix& other) {
    if (&other == this)
      return *this;
    n_rows_ = other.Rows();
    n_cols_ = other.Cols();
    nnz_ = other.NonZeroEst();
    if (other.OuterDim() > OuterDim()) {
      delete[] outer_idx_;
      outer_idx_ = new uint[other.OuterDim()];
    }
    memcpy(outer_idx_, other.outer_idx_,
           sizeof(uint) * other.OuterDim());
    other.outer_idx_ = nullptr;
    storage_ = other.Storage();
  }

  SparseMatrix(SparseMatrix &&other) noexcept
      : n_rows_(other.Rows()), n_cols_(other.Cols()) {
    delete[] outer_idx_;
    nnz_ = other.nnz_;
    outer_idx_ = other.outer_idx_;
    other.outer_idx_ = nullptr;
    storage_ = std::move(other.storage_);
  }

  // Only allowed for dense vector. Constructor for an n-dim dense vector
  explicit SparseMatrix(uint n) {
    static_assert(IsStaticVector<SparseMatrix>::value);
    data_ = new Real[n];
    if constexpr (nRows == 1)
      n_cols_ = n;
    else
      n_rows_ = n;
  }

  void Resize(uint m, uint n) {
    uint outer_dim = Major == RowMajor ? m : n;
    if (OuterDim() < outer_dim) {
      uint *new_outer_idx = new uint[outer_dim + 1];
      memcpy(new_outer_idx, outer_idx_, sizeof(uint) * (OuterDim() + 1));
      delete[] outer_idx_;
      outer_idx_ = new_outer_idx;
    }
    n_rows_ = m;
    n_cols_ = n;
  }

  Real operator[](uint i) const {
    static_assert(storageType == Dense);
    return data_[i];
  }

  Real &operator[](uint i) {
    static_assert(storageType == Dense);
    return data_[i];
  }

  using type = SparseMatrix<nRows, nCols, storageType, RowMajor>;
  using TransposedMatrix =
      SparseMatrix<nCols, nRows, storageType, transpose_op<Major>::value>;

  type Eval() const { return type(*this); }

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
  template <typename InputIterator> void SetByIterator(InputIterator& it) {
    if constexpr (storageType == Sparse) {
      uint outer_cnt = 0, inner_cnt = 0;
      while (it()) {
        while (outer_cnt < it.Outer())
          outer_idx_[++outer_cnt] = inner_cnt;
        storage_.InnerIdx(inner_cnt) = it.Inner();
        storage_.Data(inner_cnt) = it.value();
        inner_cnt++;
        ++it;
      }
      outer_idx_[outer_cnt] = inner_cnt;
      nnz_ = inner_cnt;
    } else if (storageType == Dense) {
      while (it()) {
        AccessByMajor(it.Outer(), it.Inner()) = it.value();
        ++it;
      }
    }
  }

  TransposedMatrix Transposed() const {
    TransposedMatrix ret(n_cols_, n_rows_, storage_);
    if constexpr (Major == RowMajor)
      memcpy(ret.outer_idx_, outer_idx_, sizeof(uint) * n_rows_);
    else if constexpr (Major == ColMajor)
      memcpy(ret.outer_idx_, outer_idx_, sizeof(uint) * n_cols_);
    return TransposedMatrix(n_cols_, n_rows_, storage_);
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
    int idx = IndexAt(i, j);
    return idx < 0 ? 0 : Storage().Data(static_cast<uint>(i));
  }

  Real &operator()(uint i, uint j) {
    int idx = IndexAt(i, j);
    if (idx < 0) {
      std::cerr << "Warning: reference a zero element is not allowed."
                << std::endl;
      exit(-1);
    }
    return Data(static_cast<uint>(idx));
  }

  uint OuterDim() const {
    if constexpr (Major == RowMajor)
      return Rows();
    else
      return Cols();
  }
  uint InnerDim() const {
    if constexpr (Major == RowMajor)
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
  uint *OuterIndices() const { return outer_idx_; }
  uint *InnerIndices() const { return storage_.InnerIndices(); }
  uint OuterIdx(uint i) const { return outer_idx_[i]; }
  uint InnerIdx(uint i) const { return storage_.InnerIdx(i); }
  uint Data(uint i) const { return storage_.Data(i); }
  Real *Datas() const { return storage_.Datas(); }
  int IndexAt(uint i, uint j) const {
    uint idx = storage_.SearchIndex(OuterIdx(), j);
    return idx == OuterIdx() ? -1 : static_cast<int>(idx);
  }

  void SetFromStorage(const SparseStorage &sparse_storage) {
    storage_ = sparse_storage;
  }
  class InnerIterator;
  class NonZeroIterator {
  public:
    explicit NonZeroIterator(const SparseMatrix &spm)
        : outer_idx_(spm.OuterIndices()), storage_(spm.Storage()) {}
    uint Row() const {
      if constexpr (Major == RowMajor)
        return Outer();
      else
        return Inner();
    }
    uint Col() const {
      if constexpr (Major == RowMajor)
        return Inner();
      else
        return Outer();
    }
    uint Outer() const { return outer_ptr_; }
    uint Inner() const { return storage_.InnerIdx(inner_ptr_); }
    Real value() const { return storage_.Data(inner_ptr_); }
    bool operator()() const { return inner_ptr_ < storage_.UsedSize(); }
    NonZeroIterator &operator++() {
      inner_ptr_++;
      while (inner_ptr_ > outer_idx_[outer_ptr_])
        outer_ptr_++;
      return *this;
    }

  private:
    uint inner_ptr_ = 0, outer_ptr_ = 0;
    uint *outer_idx_;
    const SparseStorage &storage_;
  };

  friend std::ostream &operator<<(std::ostream &o, const SparseMatrix &spm) {
    NonZeroIterator it(spm);
    while (it()) {
      o << it.Row() << "," << it.Col() << ", " << it.value() << std::endl;
      ++it;
    }
    return o;
  }
protected:
  uint *outer_idx_ = nullptr;
  Real *data_ = nullptr;
  compile_time_uint_t<nRows> n_rows_ = nRows;
  compile_time_uint_t<nCols> n_cols_ = nCols;
  SparseStorage storage_;
  uint nnz_ = 0;
};

using SparseMatrixXd = SparseMatrix<0, 0, Sparse, RowMajor>;

template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix<nRows, nCols, storage,
                   Major>::InnerIterator { // mainly used for multiplication
public:
  InnerIterator(const SparseMatrix &mat, uint outer_idx)
      : outer_idx_(outer_idx), begin_(mat.OuterIdx(outer_idx)),
        end_(mat.OuterIdx(outer_idx + 1)), storage_(mat.Storage()),
        cur_(mat.OuterIdx(outer_idx)) {}
  uint OuterIndex() const { return outer_idx_; }
  uint Index() const { return storage_.InnerIdx(cur_); }
  Real Data() const { return storage_.Data(cur_); }
  void Reset() { cur_ = begin_; }
  bool operator()() { return cur_ != end_; }
  InnerIterator &operator++() {
    cur_++;
    return *this;
  }
  template <typename Rhs> AddExpr<SparseMatrix, Rhs> operator+(const Rhs &rhs) {
    return AddExpr<SparseMatrix, Rhs>(rhs);
  };

private:
  uint outer_idx_{};
  uint cur_;
  uint begin_;
  uint end_;
  const SparseStorage &storage_;
};

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
  ~TripletSparseMatrix() {}

private:
  uint nnz_ = 0;
  compile_time_uint_t<nRows> n_rows_ = nRows;
  compile_time_uint_t<nRows> n_cols_ = nCols;
  uint *outer_idx_ = nullptr;
  uint *inner_idx_ = nullptr;
  Real *data_ = nullptr;
};

template <uint nRows_, uint nCols_, StorageType Storage_, StorageMajor Major_>
struct traits<SparseMatrix<nRows_, nCols_, Storage_, Major_>> {
  static constexpr uint nRows = nRows_;
  static constexpr uint nCols = nCols_;
  static constexpr StorageType storage = Storage_;
  static constexpr StorageMajor major = Major_;
};

template <uint nRows_, uint nCols_, StorageMajor Major_>
struct traits<TripletSparseMatrix<nRows_, nCols_, Major_>> {
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
