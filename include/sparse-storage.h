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
  SparseStorage(uint allocate_size) : allocated_size_(allocate_size) {
    data_ = new Real[allocate_size];
  }
  SparseStorage(uint allocate_size, uint nnz) {
    assert(allocate_size >= nnz);
    allocated_size_ = allocate_size;
    used_size_ = nnz;
    data_ = new Real[allocate_size];
    inner_idx_ = new uint[allocate_size];
  }
  explicit SparseStorage(SparseStorage&& other)
      : allocated_size_(other.allocated_size_), used_size_(other.used_size_) {
    inner_idx_ = std::move(other.inner_idx_);
    data_ = std::move(other.data_);
    other.inner_idx_ = nullptr;
    other.data_ = nullptr;
  }
  SparseStorage &operator=(SparseStorage&& other) {
    allocated_size_ = other.allocated_size_;
    used_size_ = other.used_size_;
    inner_idx_ = std::move(other.inner_idx_);
    data_ = std::move(other.data_);
    other.inner_idx_ = nullptr;
    other.data_ = nullptr;
    return *this;
  }

  ~SparseStorage() {
    delete[] inner_idx_;
    delete[] data_;
  }
  void Realloc(uint realloc_size) {
    if (realloc_size != allocated_size_) {
      uint *new_inner_idx = new uint[realloc_size];
      Real *new_data = new Real[realloc_size];
      memcpy(new_inner_idx, inner_idx_, sizeof(uint) * allocated_size_);
      memcpy(new_data, data_, sizeof(Real) * allocated_size_);
      delete[] inner_idx_;
      delete[] data_;
      inner_idx_ = new_inner_idx;
      data_ = new_data;
    }
    allocated_size_ = realloc_size;
    if (used_size_ >= allocated_size_)
      used_size_ = allocated_size_;
  }
  uint *InnerIndices() const { return inner_idx_; }
  uint InnerIdx(uint i) const { return inner_idx_[i]; }
  uint &InnerIdx(uint i) { return inner_idx_[i]; }
  Real *Datas() const { return data_; }
  Real Data(uint i) const { return data_[i]; }
  Real &Data(uint i) { return data_[i]; }

private:
  uint allocated_size_ = 0;
  uint used_size_ = 0;
  uint *inner_idx_ = nullptr;
  Real *data_ = nullptr;
};

} // namespace spmx

#endif // SPMX_SPARSE_STORAGE_H
