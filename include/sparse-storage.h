//
// Created by creeper on 23-5-16.
//

#ifndef SPMX_SPARSE_STORAGE_H
#define SPMX_SPARSE_STORAGE_H

#include <cassert>
#include <cstring>
#include <iostream>
#include <spmx-types.h>

namespace spmx {

class SparseStorage {
public:
  SparseStorage() = default;
  explicit SparseStorage(uint allocate_size) : allocated_size_(allocate_size) {
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(SparseStorage, allocate_size);
#endif
    if (!allocate_size) return ;
    inner_idx_ = new uint[allocate_size];
    data_ = new Real[allocate_size];
  }
  SparseStorage(uint allocate_size, uint nnz) {
    assert(allocate_size >= nnz);
    allocated_size_ = allocate_size;
    used_size_ = nnz;
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(SparseStorage, allocate_size);
#endif
    data_ = new Real[allocate_size];
    inner_idx_ = new uint[allocate_size];
  }

  SparseStorage(const SparseStorage &other)
      : allocated_size_(other.allocated_size_), used_size_(other.used_size_) {
    if (!other.allocated_size_)
      return;
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(SparseStorage, allocated_size_);
#endif
    data_ = new Real[allocated_size_];
    memcpy(data_, other.data_, sizeof(Real) * allocated_size_);
    inner_idx_ = new uint[allocated_size_];
    memcpy(inner_idx_, other.inner_idx_, sizeof(uint) * allocated_size_);
  }

  SparseStorage(SparseStorage &&other) noexcept
      : allocated_size_(other.allocated_size_), used_size_(other.used_size_) {
    inner_idx_ = other.inner_idx_;
    data_ = other.data_;
    other.inner_idx_ = nullptr;
    other.data_ = nullptr;
  }

  SparseStorage &operator=(const SparseStorage &other) {
    if (&other == this)
      return *this;
    used_size_ = other.used_size_;
    if (allocated_size_ == other.allocated_size_) {
      if (other.allocated_size_) {
        memcpy(inner_idx_, other.inner_idx_, sizeof(uint) * other.used_size_);
        memcpy(data_, other.data_, sizeof(Real) * other.used_size_);
      }
      return *this;
    }
#ifdef MEMORY_TRACING
    MEMORY_LOG_DELETE(SparseStorage, allocated_size_);
#endif
    allocated_size_ = other.allocated_size_;
    delete[] inner_idx_;
    delete[] data_;
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(SparseStorage, other.allocated_size_);
#endif
    inner_idx_ = new uint[other.allocated_size_];
    data_ = new Real[other.allocated_size_];
    if (other.allocated_size_) {
      memcpy(inner_idx_, other.inner_idx_, sizeof(uint) * other.used_size_);
      memcpy(data_, other.data_, sizeof(Real) * other.used_size_);
    }
    return *this;
  }

  SparseStorage &operator=(SparseStorage &&other) noexcept {
    delete[] inner_idx_;
    delete[] data_;
#ifdef MEMORY_TRACING
    MEMORY_LOG_DELETE(SparseStorage, allocated_size_);
#endif
    allocated_size_ = other.allocated_size_;
    used_size_ = other.used_size_;
    inner_idx_ = other.inner_idx_;
    data_ = other.data_;
    other.inner_idx_ = nullptr;
    other.data_ = nullptr;
    return *this;
  }

  ~SparseStorage() {
#ifdef MEMORY_TRACING
    if (inner_idx_ == nullptr && data_ == nullptr)
      MEMORY_LOG_MESSAGE(SparseStorage, "Deleting nullptr");
    else if (inner_idx_ == nullptr || data_ == nullptr)
      MEMORY_LOG_MESSAGE(SparseStorage, "Warning: SparseStorage is partially put to nullptr");
    else
      MEMORY_LOG_DELETE(SparseStorage, allocated_size_);
#endif
    delete[] inner_idx_;
    delete[] data_;
  }

  void Realloc(uint realloc_size) {
    if (realloc_size != allocated_size_) {
      uint *new_inner_idx = new uint[realloc_size];
      Real *new_data = new Real[realloc_size];
#ifdef MEMORY_TRACING
      MEMORY_LOG_ALLOC(SparseStorage, realloc_size);
#endif
      if (allocated_size_) {
        memcpy(new_inner_idx, inner_idx_, sizeof(uint) * allocated_size_);
        memcpy(new_data, data_, sizeof(Real) * allocated_size_);
      }
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseStorage, allocated_size_);
#endif
      delete[] inner_idx_;
      delete[] data_;
      inner_idx_ = new_inner_idx;
      data_ = new_data;
    }
    allocated_size_ = realloc_size;
    if (used_size_ >= allocated_size_)
      used_size_ = allocated_size_;
  }

  uint SearchIndex(uint l, uint r, uint idx) const {
    while (l <= r) {
      uint mid = (l + r) >> 1;
      if (inner_idx_[mid] < idx)
        r = mid - 1;
      else if (inner_idx_[mid] == idx)
        return static_cast<int>(mid);
      else
        l = mid + 1;
    }
  }

  void Reserve(uint size) {
    if (size <= allocated_size_)
      return;
    uint *new_inner_idx = new uint[size];
    Real *new_data = new Real[size];
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(SparseStorage, size);
#endif
    if (allocated_size_) {
      memcpy(new_inner_idx, inner_idx_, sizeof(uint) * allocated_size_);
      memcpy(new_data, data_, sizeof(Real) * allocated_size_);
#ifdef MEMORY_TRACING
      MEMORY_LOG_DELETE(SparseStorage, allocated_size_);
#endif
      delete[] inner_idx_;
      delete[] data_;
    }
    allocated_size_ = size;
    inner_idx_ = new_inner_idx;
    data_ = new_data;
  }
  void Reset(uint size) {
    memset(data_, 0, sizeof(Real) * size);
  }
  uint *InnerIndices() const { return inner_idx_; }
  uint InnerIdx(uint i) const {
#ifdef MEMORY_TRACING
    if (i >= allocated_size_) {
      MEMORY_LOG_INVALID_ACCESS(SparseStorage, i);
      exit(-1);
    }
#endif
     return inner_idx_[i];
  }
  uint &InnerIdx(uint i) { return inner_idx_[i]; }
  Real *Datas() const { return data_; }
  Real Data(uint i) const {
#ifdef MEMORY_TRACING
     if (i >= allocated_size_) {
      MEMORY_LOG_INVALID_ACCESS(SparseStorage, i);
      exit(-1);
     }
#endif
     return data_[i];
  }
  Real &Data(uint i) { return data_[i]; }
  uint UsedSize() const { return used_size_; }
  void SetUsed(uint nnz) { used_size_ = nnz; }
  uint AllocatedSize() const { return allocated_size_; }

private:
  uint allocated_size_ = 0;
  uint used_size_ = 0;
  uint *inner_idx_ = nullptr;
  Real *data_ = nullptr;
};

} // namespace spmx

#endif // SPMX_SPARSE_STORAGE_H
