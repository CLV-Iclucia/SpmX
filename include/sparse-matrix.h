//
// Created by creeper on 23-2-23.
//

#ifndef SPMX_SPARSE_MATRIX_H
#define SPMX_SPARSE_MATRIX_H

#include "my-stl.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <sparse-storage.h>
#include <spmx-options.h>
#include <spmx-types.h>
#include <spmx-utils.h>
#include <type-utils.h>
#include <utility>

namespace spmx {

template <uint nRows, uint nCols, StorageType Storage, StorageMajor Major>
class SparseMatrix;

template <typename Lhs, typename Rhs>
class LinearExpr;
/**
 * I use CRTP for all the matrix and vectors!
 * Use this we can hold sparse matrices in various forms.
 * @tparam Derived
 */
template <typename Derived> class SparseMatrixBase {
public:
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<Derived *>(this); }
  inline uint OuterDim() const { return derived().OuterDim(); }
  inline uint InnerDim() const { return derived().InnerDim(); }
  inline uint OuterIdx(uint i) const { return derived().OuterIdx(i); }
  inline uint InnerIdx(uint i) const { return derived().InnerIdx(i); }
  inline Real Data(uint i) const { return derived().Data(i); }

  inline uint Rows() const { return derived().Rows(); }
  inline uint Cols() const { return derived().Cols(); }
  inline uint NonZeroEst() const { return derived().NonZeroEst(); }

  bool IsVector() const { return Rows() == 1 || Cols() == 1; }

  template <typename Lhs, typename Rhs>
  using ProdRet = typename ProductReturnType<Lhs, Rhs>::type;

  template <typename Lhs, typename Rhs>
  using SumRet = typename SumReturnType<Lhs, Rhs>::type;

  using Lhs = Derived;
  template <typename Rhs>
  ProdRet<Lhs, Rhs> operator*(const SparseMatrixBase<Rhs> &rhs) const {
    if constexpr (Major != traits<Rhs>::StorageMajor) {
      uint est_nnz = 0;
      for (uint i = 0; i < OuterDim(); i++) {
        for (uint j = 0; j < rhs.OuterDim(); j++) {
          uint lptr = OuterIdx(i), rptr = rhs.OuterIdx(j);
          while (lptr < InnerIdx(i + 1) && rptr < rhs.InnerIdx(j + 1)) {
            if (InnerIdx(lptr) < rhs.InnerIdx(rptr))
              lptr++;
            else if (rhs.InnerIdx(rptr) < InnerIdx(lptr))
              rptr++;
            else {
              est_nnz++;
              break;
            }
          }
        }
      }
      ProdRet<Lhs, Rhs> ret(Rows(), Cols(), est_nnz);
      uint cnt = 0;
      for (uint i = 0; i < OuterDim(); i++) {
        ret.OuterIdx(i) = cnt;
        for (uint j = 0; j < rhs.OuterDim(); j++) {
          uint lptr = OuterIdx(i), rptr = rhs.OuterIdx(j);
          Real sum = 0.0;
          while (lptr < OuterIdx(i + 1) && rptr < rhs.OuterIdx(j + 1)) {
            if (InnerIdx(lptr) < rhs.InnerIdx(rptr))
              lptr++;
            else if (InnerIdx(lptr) > rhs.InnerIdx(rptr))
              rptr++;
            else {
              sum += Data(lptr) * rhs.Data(rptr);
              lptr++;
              rptr++;
            }
          }
          if (!IsZero(sum)) {
            ret.InnerIdx(cnt) = j;
            ret.Data(cnt++) = sum;
          }
        }
        ret.OuterIdx(ret.OuterDim()) = cnt;
      }
      return ret;
    } else {
      uint est_nnz = 0;
      // estimate the number of non-zero elements of the result
      BitSet bucket(InnerDim());
      for (uint i = 0; i < OuterDim(); i++) {
        bucket.Clear();
        for (uint j = OuterIdx(i); j < OuterIdx(i + 1); j++) {
          uint idx = InnerIdx(j);
          for (uint k = rhs.OuterIdx(idx); k < rhs.OuterIdx(idx + 1); k++)
            if (!bucket(rhs.InnerIdx(k)))
              bucket.Set(rhs.InnerIdx(k));
        }
        est_nnz += bucket.BitCnt();
      }
      ProdRet<Lhs, Rhs> ret(Rows(), rhs.Cols(), est_nnz);
      uint cnt = 0;
      for (uint i = 0; i < OuterDim(); i++) { // TODO: this is wrong
        bucket.Clear();
        ret.OuterIdx(i) = cnt;
        for (uint j = OuterIdx(i); j < OuterIdx(i + 1); j++) {
          uint idx = InnerIdx(j);
          for (uint k = rhs.OuterIdx(idx); k < rhs.OuterIdx(idx + 1); k++) {
            if (!bucket(rhs.InnerIdx(k))) {
              ret.inner[ret.nnz++] = rhs.InnerIdx(k);
            }
            ret.val[ret.nnz] += Data(j) * rhs.Data(k);
          }
        }
      }
      ret.OuterIdx(rhs.OuterDim()) = cnt;
      return ret;
    }
  }

  template <typename Rhs>
  LinearExpr<Lhs, Rhs> operator+(const SparseMatrixBase<Rhs> &rhs) {
    return LinearExpr<Lhs, Rhs>(1, *this, 1, rhs);
  }

  template <typename Rhs>
  LinearExpr<Lhs, Rhs> operator-(const SparseMatrixBase<Rhs> &rhs) {
    return LinearExpr<Lhs, Rhs>(1, -1, rhs);
  }

  template <typename Rhs>
  Lhs& operator+=(const SparseMatrixBase<Rhs> &rhs) {
    *this = *this + rhs;
    return *this;
  }

  template <typename Rhs>
  Lhs& operator-=(const SparseMatrixBase<Rhs> &rhs) {
    *this = *this - rhs;
    return *this;
  }

private:
  using traits<Derived>::nRows;
  using traits<Derived>::nCols;
  using traits<Derived>::Storage;
  using traits<Derived>::Major;
};

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
template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix
    : public SparseMatrixBase<SparseMatrix<nRows, nCols, storage, Major>> {
public:
  SparseMatrix() = default;
  SparseMatrix(uint n_rows, uint n_cols, uint nnz) {
    if constexpr (!nRows)
      n_rows_ = n_rows;
    if constexpr (!nCols)
      n_cols_ = n_cols;
    if (storage == Sparse) {
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
  void SetFromTriplets(Iterator begin, Iterator end) {}
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

  explicit SparseMatrix(SparseMatrixBase<SparseMatrix> &&other) noexcept
      : n_rows_(other.Rows()), n_cols_(other.Cols()) {
    outer_idx_ = std::move(other.outer_idx_);
    storage_ = std::move(other.Storage());
  }

  using type = SparseMatrix<nRows, nCols, storage, RowMajor>;
  using TransposedMatrix =
      SparseMatrix<nCols, nRows, storage, transpose_op<Major>::value>;

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
  type Eval() const { return type(*this); }

  /**
   * The following methods are only enabled for dense storage
   * @param i
   * @param j
   * @return
   */
  Real AccessByMajor(uint i, uint j) const {
    return data_[i * OuterDim() + j];
  }
  Real& AccessByMajor(uint i, uint j) {
    return data_[i * OuterDim() + j];
  }
  /**
   * Set the sparse matrix based on the input iterator
   * it runs the iterator and set the corresponding
   * for now, the iterator and the matrix must be in the same storage major
   * @tparam InputIterator
   * @param it
   */
  template<typename InputIterator>
  void SetByTriplet(InputIterator it) {
    if constexpr (storage == Sparse) {
      uint outer_cnt = 0, inner_cnt = 0;
      while(it()) {
        while(outer_cnt < it.Outer()) outer_idx_[++outer_cnt] = inner_cnt;
        storage_.InnerIdx(inner_cnt) = it.Inner();
        storage_.Data(inner_cnt) = it.value();
        it++;
      }
      outer_idx_[outer_cnt] = inner_cnt;
      nnz_ = inner_cnt;
    } else if (storage == Dense) {
      while(it()) {
        AccessByMajor(it.Outer(), it.Inner()) = it.value();
        it++;
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
    return Storage().Data(static_cast<uint>(idx));
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
  class NonZeroIterator;

protected:
  uint *outer_idx_ = nullptr;
  Real *data_ = nullptr;
  compile_time_uint_t<nRows> n_rows_ = nRows;
  compile_time_uint_t<nCols> n_cols_ = nCols;
  std::enable_if_t<storage == Sparse, SparseStorage> storage_;
  std::enable_if_t<storage == Sparse, SparseStorage> nnz_ = 0;
};

template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix<nRows, nCols, storage,
                   Major>::InnerIterator { // mainly used for multiplication
public:
  using SparseMatrix = SparseMatrix<nRows, nCols, storage, Major>;
  InnerIterator(const SparseMatrix &mat, uint outer_idx)
      : outer_idx_(outer_idx), begin_(mat.OuterIdx(outer_idx)),
        end_(mat.OuterIdx(outer_idx + 1)), storage_(mat.Storage()),
        cur_(mat.OuterIdx(outer_idx)) {}
  uint OuterIndex() const { return outer_idx_; }
  uint Index() const { return storage_.InnerIdx(cur_); }
  Real Data() const { return storage_.Data(cur_); }
  void Reset() { cur_ = begin_; }
  bool NotEnd() { return cur_ != end_; }
  InnerIterator &operator++() {
    cur_++;
    return *this;
  }
  template<typename Rhs>
  LinearExpr<SparseMatrix, Rhs> operator+(const Rhs& rhs){
      return LinearExpr<SparseMatrix, Rhs>(rhs);
  };
private:
  uint outer_idx_{};
  uint cur_;
  uint begin_;
  uint end_;
  const SparseStorage &storage_;
};

template <uint nRows, uint nCols, StorageType storage, StorageMajor Major>
class SparseMatrix<nRows, nCols, storage,
                   Major>::NonZeroIterator { // mainly used for multiplication
public:
  using SparseMatrix = SparseMatrix<nRows, nCols, storage, Major>;
  explicit NonZeroIterator(const SparseMatrix &spm)
      : storage_(spm.Storage()), outer_idx_(spm.OuterIndices()) {}
  uint Outer() const { return outer_ptr_; }
  uint Inner() const { return storage_.InnerIdx(inner_ptr_); }
  Real value() const { return storage_.Data(inner_ptr_); }
  bool operator()() const { return inner_ptr_ < storage_.UsedSize(); }
  InnerIterator &operator++() {
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
  uint *outer_idx_;
  uint *inner_idx_;
  Real *data_;
};

template <uint nRows_, uint nCols_, StorageType Storage_, StorageMajor Major_>
struct traits<SparseMatrix<nRows_, nCols_, Storage_, Major_>> {
  static constexpr uint nRows = nRows_;
  static constexpr uint nCols = nCols_;
  static constexpr StorageType Storage = Storage_;
  static constexpr StorageMajor major = Major_;
};

template <uint nRows_, uint nCols_, StorageMajor Major_>
struct traits<TripletSparseMatrix<nRows_, nCols_, Major_>> {
  static constexpr uint nRows = nRows_;
  static constexpr uint nCols = nCols_;
  static constexpr StorageType Storage = Sparse;
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
