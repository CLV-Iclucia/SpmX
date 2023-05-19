//
// Created by creeper on 23-3-24.
//

#ifndef SPMX_SPMX_VECTOR_H
#define SPMX_SPMX_VECTOR_H

#include "spmx-utils.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <spmx-types.h>
#include <stdexcept>
#include <utility>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include <my-tiny-blas.h>
#include <sparse-storage.h>

namespace spmx {
template <VectorStorage Storage> class Vector;

template <> class Vector<Dense> {
private:
  unsigned int dim_ = 0;
  Real *data_ = nullptr;

public:
  void RandFill() {
    for (uint i = 0; i < dim_; i++)
      data_[i] = RandReal();
  }
  static Vector RandVec() {
    uint _n = Randu();
    Vector vec(_n);
    vec.RandFill();
    return vec;
  }
  static Vector RandVec(uint _n) {
    Vector vec(_n);
    vec.RandFill();
    return vec;
  }
  Vector() = default;
  Vector(unsigned int _n, const Real *A) : dim_(_n) {
    data_ = new Real[dim_];
    memcpy(data_, A, sizeof(Real) * dim_);
  }
  explicit Vector(unsigned int _n) : dim_(_n) {
    data_ = new Real[dim_];
    memset(data_, 0, sizeof(Real) * dim_);
  }
  Vector(const Vector &A) {
    dim_ = A.dim_;
    data_ = new Real[dim_];
    memcpy(data_, A.data_, sizeof(Real) * dim_);
  }
  Vector(Vector &&A) noexcept {
    dim_ = A.dim_;
    data_ = A.data_;
    A.data_ = nullptr;
  }
  Vector &operator=(const Vector &A) {
    if (&A == this)
      return *this;
    else {
      if (dim_ != A.dim_) {
        dim_ = A.dim_;
        delete[] data_;
        data_ = new Real[dim_];
      }
      memcpy(data_, A.data_, sizeof(Real) * dim_);
      return *this;
    }
  }
  Vector &operator=(Vector &&A) noexcept {
    if (dim_ != A.dim_) {
      dim_ = A.dim_;
      delete[] data_;
      data_ = A.data_;
      A.data_ = nullptr;
    } else
      memcpy(data_, A.data_, sizeof(Real) * dim_);
    return *this;
  }
  void fill(Real val) {
    for (uint i = 0; i < dim_; i++)
      data_[i] = val;
  }

  template<VectorStorage StorageRHS>
  Real dot(const Vector<StorageRHS> &A) const {
    if constexpr (StorageRHS == Dense)
      return ds_dot(data_, A.data_, dim_);
    else if constexpr (StorageRHS == Sparse)
      return ds_sp_dot(data_, A.idx_, A.data_, dim_, A.nnz_);
  }
  void resize(int n) {
    dim_ = n;
    delete[] data_;
    data_ = new Real[dim_];
  }
  Real &operator[](int i) { return data_[i]; }
  const Real &operator[](int i) const { return data_[i]; }

  template<VectorStorage StorageRHS>
  void saxpy(const Vector<StorageRHS> &A, Real k) {
    if constexpr (StorageRHS == Dense) {
      Vector vec(dim_);
      ds_saxpy(data_, A.data_, vec.data_, k, dim_);
    }
    else if constexpr (StorageRHS == Sparse) {

    }
  }
  void scadd(const Vector &A, Real k) {
    int i = 0;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] = k * data_[i] + A.data_[i];
  }
  template<VectorStorage StorageRHS>
  Vector operator+(const Vector<StorageRHS> &A) const {
    if constexpr (StorageRHS == Dense) {
      Vector vec(dim_);
      ds_add(data_, A.data_, vec.data_, dim_);
    }
    else if constexpr (StorageRHS == Sparse) {
      Vector vec(dim_);
      ds_sp_add(data_, A.idx_, A.data_, vec.data_, dim_, A.nnz_);
    }
  }
  Vector operator-() {
    Vector V(dim_);
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      V.data_[i] = -data_[i];
    return V;
  }
  Vector operator-(const Vector &A) const {
    Vector V(dim_);
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      V.data_[i] = data_[i] - A.data_[i];
    return V;
  }
  Vector operator/(Real val) const {
    if (val == 0) {
      std::printf("Division by zero!");
      exit(-1);
    }
    Vector V(dim_);
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      V.data_[i] = data_[i] / val;
    return V;
  }
  Vector operator*(Real val) const {
    Vector V(dim_);
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      V.data_[i] = val * data_[i];
    return V;
  }
  friend Vector operator*(Real val, const Vector &A) { return A * val; }
  Vector &operator*=(const Vector &A) {
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] *= A.data_[i];
    return *this;
  }
  Vector &inv() {
    for (uint i = 0; i < dim_; i++)
      data_[i] = 1.0 / data_[i];
    return *this;
  }
  Vector inv() const {
    Vector ret(dim_);
    for (uint i = 0; i < dim_; i++)
      ret[i] = 1.0 / data_[i];
    return ret;
  }
  Vector &operator+=(const Vector &A) {
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] += A.data_[i];
    return *this;
  }
  Vector &operator/=(const Vector &A) {
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] /= A.data_[i];
    return *this;
  }
  Vector &operator/=(Real val) {
    if (val == 0) {
      std::printf("Division by zero!");
      exit(-1);
    }
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] /= val;
    return *this;
  }
  Vector &operator-=(const Vector &A) {
    int i;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] -= A.data_[i];
    return *this;
  }
  Vector &operator*=(Real val) {
    for (int i = 0; i < dim_; i++)
      data_[i] *= val;
    return *this;
  }
  uint dim() const { return dim_; }
  Real L2Norm() const {
    assert(dim_ > 0);
    Real maxv = std::abs(data_[0]), sum = 0.0;
    for (uint i = 1; i < dim_; i++)
      maxv = std::max(std::abs(data_[i]), maxv);
    for (uint i = 0; i < dim_; i++)
      sum += (data_[i] / maxv) * (data_[i] / maxv);
    return std::sqrt(sum) * maxv;
  }
  Real L2NormSqr() const {
    assert(dim_ > 0);
    Real maxv = std::abs(data_[0]), sum = 0.0;
    for (uint i = 1; i < dim_; i++)
      maxv = std::max(std::abs(data_[i]), maxv);
    for (uint i = 0; i < dim_; i++)
      sum += (data_[i] / maxv) * (data_[i] / maxv);
    return sum * maxv * maxv;
  }
  Real L1Norm() const {
    assert(dim_ > 0);
    Real sum = 0.0;
    for (uint i = 0; i < dim_; i++)
      sum += std::abs(data_[i]);
    return sum;
  }
  Vector Normalized() const {
    Real norm = L2Norm();
    return (*this) / norm;
  }
  friend std::ostream &operator<<(std::ostream &o, const Vector &A) {
    o << A.data_[0];
    for (uint i = 1; i < A.dim_; i++)
      o << " " << A.data_[i];
    return o;
  }
  ~Vector() { delete[] data_; }
};
template <> class Vector<Sparse> {
private:
  unsigned int dim_ = 0;
  unsigned int nnz_ = 0;
  SparseStorage<Dynamic> storage_;

public:
  Vector() = default;
  explicit Vector(uint n) : dim_(n) {}
  explicit Vector(uint n, uint nnz) : dim_(n), nnz_(nnz) {
  }
  Vector(const Vector<Sparse> &A) : dim_(A.dim_), nnz_(A.nnz_) {
  }
  Vector(Vector &&A) noexcept {
    dim_ = A.dim_;
    storage_ = A.storage_;
  }
  Vector &operator=(const Vector &A) {
    if (&A == this)
      return *this;
    else {
      dim_ = A.dim_;
      storage_ = A.storage_;
      return *this;
    }
  }
  Vector &operator=(Vector &&A) noexcept {
    dim_ = A.dim_;
    storage_ = A.storage_;
    return *this;
  }
  Real dot(const Vector &A) const {
    assert(dim_ == A.dim_);
    Real sum = 0.0;
    uint lptr = 0, rptr = 0;
    while (lptr < nnz_ && rptr < A.nnz_) {
      if (data[lptr].idx < A.data[rptr].idx)
        lptr++;
      else if (data[lptr].idx > A.data[rptr].idx)
        rptr++;
      else
        sum += data[lptr].val * A.data[rptr].val;
    }
    return sum;
  }
  void saxpy(const Vector &B, Real k) {
    assert(dim_ == B.dim_);
    uint lptr = 0, rptr = 0;
    uint _nnz = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        lptr++;
      else if (data[lptr].idx > B.data[rptr].idx)
        rptr++;
      else {
        lptr++;
        rptr++;
      }
      _nnz++;
    }
    _nnz += nnz_ - lptr + B.nnz_ - rptr;
    Pair *new_data = new Pair[_nnz];
    lptr = rptr = 0;
    uint ptr = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        new_data[ptr] = data[lptr++];
      else if (data[lptr].idx > B.data[rptr].idx) {
        new_data[ptr].idx = B.data[rptr].idx;
        new_data[ptr].val = k * B.data[rptr].val;
        rptr++;
      } else {
        new_data[ptr].idx = data[rptr].idx;
        new_data[ptr].val = data[lptr].val + k * B.data[rptr].val;
        lptr++;
        rptr++;
      }
      ptr++;
    }
    while (lptr < nnz_)
      new_data[ptr++] = data[lptr++];
    while (rptr < B.nnz_) {
      new_data[ptr].idx = B.data[rptr].idx;
      new_data[ptr].val = k * B.data[rptr].val;
      rptr++;
      ptr++;
    }
    delete[] data;
    data = new_data;
  }
  void scadd(const Vector &B, Real k) {
    assert(dim_ == B.dim_);
    uint lptr = 0, rptr = 0;
    uint _nnz = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        lptr++;
      else if (data[lptr].idx > B.data[rptr].idx)
        rptr++;
      else {
        lptr++;
        rptr++;
      }
      _nnz++;
    }
    _nnz += nnz_ - lptr + B.nnz_ - rptr;
    Pair *new_data = new Pair[_nnz];
    lptr = rptr = 0;
    uint ptr = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx) {
        new_data[ptr].idx = data[lptr].idx;
        new_data[ptr].val = k * data[lptr++].val;
      } else if (data[lptr].idx > B.data[rptr].idx)
        new_data[ptr] = B.data[rptr++];
      else {
        new_data[ptr].idx = data[lptr].idx;
        new_data[ptr].val = k * data[lptr].val + B.data[rptr].val;
        lptr++;
        rptr++;
      }
      ptr++;
    }
    while (lptr < nnz_) {
      new_data[ptr].idx = data[lptr].idx;
      new_data[ptr++].val = data[lptr++].val;
    }
    while (rptr < B.nnz_)
      new_data[ptr++] = B.data[rptr++];
    delete[] data;
    data = new_data;
  }
  Vector operator+(const Vector &B) const {
    assert(dim_ == B.dim_);
    uint lptr = 0, rptr = 0;
    uint _nnz = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        lptr++;
      else if (data[lptr].idx > B.data[rptr].idx)
        rptr++;
      else {
        lptr++;
        rptr++;
      }
      _nnz++;
    }
    _nnz += nnz_ - lptr + B.nnz_ - rptr;
    Pair *new_data = new Pair[_nnz];
    Vector ret(_nnz);
    lptr = rptr = 0;
    uint ptr = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        new_data[ptr] = data[lptr++];
      else if (data[lptr].idx > B.data[rptr].idx)
        new_data[ptr] = B.data[rptr++];
      else {
        new_data[ptr].idx = data[lptr].idx;
        new_data[ptr].val = data[lptr].val + B.data[rptr].val;
        lptr++;
        rptr++;
      }
      ptr++;
    }
    while (lptr < nnz_) {
      new_data[ptr].idx = data[lptr].idx;
      new_data[ptr++].val = data[lptr++].val;
    }
    while (rptr < B.nnz_)
      new_data[ptr++] = B.data[rptr++];
    ret.data = new_data;
    return ret;
  }
  Vector operator-() {
    Vector ret(*this);
    for (int i = 0; i < nnz_; i++)
      ret.data[i].val = -ret.data[i].val;
    return ret;
  }
  Vector operator-(const Vector &B) const {
    assert(dim_ == B.dim_);
    uint lptr = 0, rptr = 0;
    uint _nnz = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        lptr++;
      else if (data[lptr].idx > B.data[rptr].idx)
        rptr++;
      else {
        lptr++;
        rptr++;
      }
      _nnz++;
    }
    _nnz += nnz_ - lptr + B.nnz_ - rptr;
    Pair *new_data = new Pair[_nnz];
    Vector ret(_nnz);
    lptr = rptr = 0;
    uint ptr = 0;
    while (lptr < nnz_ && rptr < B.nnz_) {
      if (data[lptr].idx < B.data[rptr].idx)
        new_data[ptr] = data[lptr++];
      else if (data[lptr].idx > B.data[rptr].idx) {
        new_data[ptr].idx = B.data[rptr].idx;
        new_data[ptr].val = -B.data[rptr++].val;
        ptr++;
      } else {
        new_data[ptr].idx = data[lptr].idx;
        new_data[ptr].val = data[lptr].val - B.data[rptr].val;
        lptr++;
        rptr++;
      }
      ptr++;
    }
    while (lptr < nnz_) {
      new_data[ptr].idx = data[lptr].idx;
      new_data[ptr++].val = data[lptr++].val;
    }
    while (rptr < B.nnz_) {
      new_data[ptr].idx = B.data[rptr].idx;
      new_data[ptr++].val = -B.data[rptr++].val;
    }
    ret.data = new_data;
    return ret;
  }
  Vector operator/(Real val) const {
    if (val == 0) {
      std::printf("Division by zero!");
      exit(-1);
    }
    Vector V(dim_, nnz_);
    for (int i = 0; i < nnz_; i++)
      V.data[i].val = data[i].val * val;
    return V;
  }
  Vector operator*(Real val) const {
    Vector V(dim_, nnz_);
    for (int i = 0; i < nnz_; i++)
      V.data[i].val = data[i].val * val;
    return V;
  }
  friend Vector operator*(Real val, const Vector &A) { return A * val; }
  Vector &inv() {
    for (int i = 0; i < nnz_; i++)
      data[i].val = 1.0 / data[i].val;
    return *this;
  }
  Vector inv() const {
    Vector ret(*this);
    for (int i = 0; i < nnz_; i++)
      ret.data[i].val = 1.0 / ret.data[i].val;
    return ret;
  }
  Vector &operator+=(const Vector &A) {
    saxpy(A, 1.0);
    return *this;
  }
  Vector &operator/=(Real val) {
    if (val == 0) {
      std::printf("Division by zero!");
      exit(-1);
    }
    int i;
    // #pragma omp parallel for
    for (i = 0; i < nnz_; i++)
      data[i].val /= val;
    return *this;
  }
  Vector &operator-=(const Vector<Sparse> &A) {
    saxpy(A, -1.0);
    return *this;
  }
  Vector &operator*=(Real val) {
    for (int i = 0; i < nnz_; i++)
      data[i].val *= val;
    return *this;
  }
  uint dim() const { return dim_; }
  uint nonZeros() const { return nnz_; }
  Real L2Norm() const {
    assert(dim_ > 0);
    if (!nnz_)
      return 0.0;
    Real maxv = std::abs(data[0].val), sum = 0.0;
    for (uint i = 1; i < nnz_; i++)
      maxv = std::max(std::abs(data[i].val), maxv);
    for (uint i = 0; i < nnz_; i++)
      sum += (data[i].val / maxv) * (data[i].val / maxv);
    return std::sqrt(sum) * maxv;
  }
  Real L2NormSqr() const {
    assert(dim_ > 0);
    if (!nnz_)
      return 0.0;
    Real maxv = std::abs(data[0].val), sum = 0.0;
    for (uint i = 1; i < dim_; i++)
      maxv = std::max(std::abs(data[i].val), maxv);
    for (uint i = 0; i < dim_; i++)
      sum += (data[i].val / maxv) * (data[i].val / maxv);
    return sum * maxv * maxv;
  }
  Real L1Norm() const {
    assert(dim_ > 0);
    Real sum = 0.0;
    for (uint i = 0; i < nnz_; i++)
      sum += std::abs(data[i].val);
    return sum;
  }
  Vector normalized() const {
    Real norm = L2Norm();
    Vector ret(dim_);
    for (int i = 0; i < nnz_; i++)
      data[i].val = data[i].val / norm;
    return ret;
  }
  friend std::ostream &operator<<(std::ostream &o, const Vector &A) {
    if (A.nnz_)
      o << "(" << A.data[0].idx << ", " << A.data[0].val << ")";
    for (int i = 1; i < A.nnz_; i++)
      o << std::endl << "(" << A.data[0].idx << ", " << A.data[0].val << ")";
    return o;
  }
  ~Vector() { delete[] data; }
};
} // namespace spmx

#endif // SPMX_SPMX_VECTOR_H
