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

#include <sparse-storage.h>

namespace spmx {
template <StorageType Storage, VectorType VecType> class Vector;

template <VectorType VecType> class Vector<Dense, VecType> {
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

  template <StorageType StorageRHS, VectorType RHSType>
  Real dot(const Vector<StorageRHS, RHSType> &A) const {
    if constexpr (StorageRHS == Dense)
      return ds_dot(data_, A.data_, dim_);
    else if constexpr (StorageRHS == Sparse)
      return ds_sp_dot(data_, A.idx_, A.data_, dim_, A.nnz_);
  }
  void Resize(int n) {
    dim_ = n;
    delete[] data_;
    data_ = new Real[dim_];
  }
  Real &operator[](uint i) { return data_[i]; }
  Real operator[](uint i) const { return data_[i]; }
  Real &operator()(uint i) { return data_[i]; }
  Real operator()(uint i) const { return data_[i]; }

  template <StorageType StorageRHS, VectorType RHSType>
  void Saxpy(const Vector<StorageRHS, RHSType> &A, Real k) {
    if constexpr (StorageRHS == Dense) {
      Vector vec(dim_);
      ds_saxpy(data_, A.data_, vec.data_, k, dim_);
    } else if constexpr (StorageRHS == Sparse) {
    }
  }
  void ScaleAdd(const Vector &A, Real k) {
    int i = 0;
    // #pragma omp parallel for
    for (i = 0; i < dim_; i++)
      data_[i] = k * data_[i] + A.data_[i];
  }
  template <StorageType StorageRHS, VectorType RHSType>
  Vector operator+(const Vector<StorageRHS, RHSType> &A) const {
    if constexpr (StorageRHS == Dense) {
      Vector vec(dim_);
      ds_add(data_, A.data_, vec.data_, dim_);
    } else if constexpr (StorageRHS == Sparse) {
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
  uint Dim() const { return dim_; }
  Real RobustL2Norm() const {
    assert(dim_ > 0);
    Real maxv = std::abs(data_[0]), sum = 0.0;
    for (uint i = 1; i < dim_; i++)
      maxv = std::max(std::abs(data_[i]), maxv);
    for (uint i = 0; i < dim_; i++)
      sum += (data_[i] / maxv) * (data_[i] / maxv);
    return std::sqrt(sum) * maxv;
  }
  Real RobustL2NormSqr() const {
    assert(dim_ > 0);
    Real maxv = std::abs(data_[0]), sum = 0.0;
    for (uint i = 1; i < dim_; i++)
      maxv = std::max(std::abs(data_[i]), maxv);
    for (uint i = 0; i < dim_; i++)
      sum += (data_[i] / maxv) * (data_[i] / maxv);
    return sum * maxv * maxv;
  }
  Real L2Norm() const {
    return std::sqrt(dot(*this));
  }
  Real L2NormSqr() const {
    return dot(*this);
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

private:
  unsigned int dim_ = 0;
  Real *data_ = nullptr;
};
template <VectorType VecType> class Vector<Sparse, VecType> {
private:
  unsigned int dim_ = 0;
  unsigned int nnz_ = 0;
  SparseStorage storage_;
  /**
   * v <- alpha * v + beta * rhs
   * @param alpha
   * @param beta
   */
  template <StorageType VecStorage, VectorType RHSType>
  void LinearUpdate(Real alpha, Real beta, const Vector<VecStorage, RHSType> &rhs) {
    if constexpr (VecStorage == Sparse) {
      uint lptr = 0, rptr = 0, est_nnz = 0;
      while (lptr < nnz_ && rptr < rhs.nnz_) {
        if (Idx(lptr) < rhs.Idx(rptr))
          lptr++;
        else if (Idx(lptr) > rhs.Idx(rptr))
          rptr++;
        else {
          lptr++;
          rptr++;
        }
        est_nnz++;
      }
      est_nnz += nnz_ - lptr + rhs.nnz_ - rptr;
      SparseStorage new_storage(est_nnz);
      lptr = rptr = 0;
      uint ptr = 0;
      while (lptr < nnz_ && rptr < rhs.nnz_) {
        if (Idx(lptr) < rhs.Idx(rptr)) {
          new_storage.InnerIdx(ptr) = Idx(lptr);
          new_storage.Data(ptr) = alpha * Data(lptr);
          lptr++;
        } else if (Idx(lptr) > rhs.Idx(rptr)) {
          new_storage.InnerIdx(ptr) = rhs.Idx(rptr);
          new_storage.Data(ptr) = beta * rhs.Data(rptr);
          rptr++;
        } else {
          new_storage.InnerIdx(ptr) = rhs.Idx(rptr);
          new_storage.Data(ptr) = alpha * Data(lptr) + beta * rhs.Data(rptr);
          lptr++;
          rptr++;
        }
        ptr++;
      }
      while (lptr < nnz_) {
        new_storage.InnerIdx(ptr) = Idx(lptr);
        new_storage.Data(ptr) = alpha * Data(lptr);
        lptr++;
        ptr++;
      }
      while (rptr < rhs.nnz_) {
        new_storage.InnerIdx(ptr) = rhs.Idx(rptr);
        new_storage.Data(ptr) = beta * rhs.Data(rptr);
        rptr++;
        ptr++;
      }
      storage_ = std::move(new_storage);
    } else {
      // Implement sparse dense linear update
    }
  }
  template <StorageType VecStorage, VectorType RHSType>
  Vector<VecStorage, VecType> LinearCombine(Real alpha, Real beta,
                                   const Vector<VecStorage, RHSType> &rhs) const {
    if constexpr (VecStorage == Sparse) {
      uint lptr = 0, rptr = 0, est_nnz = 0;
      while (lptr < nnz_ && rptr < rhs.nnz_) {
        if (Idx(lptr) < rhs.Idx(rptr))
          lptr++;
        else if (Idx(lptr) > rhs.Idx(rptr))
          rptr++;
        else {
          lptr++;
          rptr++;
        }
        est_nnz++;
      }
      est_nnz += nnz_ - lptr + rhs.nnz_ - rptr;
      Vector<Sparse, VecType> ret(est_nnz);
      lptr = rptr = 0;
      uint ptr = 0;
      while (lptr < nnz_ && rptr < rhs.nnz_) {
        if (Idx(lptr) < rhs.Idx(rptr)) {
          ret.Idx(ptr) = Idx(lptr);
          ret.Data(ptr) = alpha * Data(lptr);
          lptr++;
        } else if (Idx(lptr) > rhs.Idx(rptr)) {
          ret.Idx(ptr) = rhs.Idx(rptr);
          ret.Data(ptr) = beta * rhs.Data(rptr);
          rptr++;
        } else {
          ret.Idx(ptr) = rhs.Idx(rptr);
          ret.Data(ptr) = alpha * Data(lptr) + beta * rhs.Data(rptr);
          lptr++;
          rptr++;
        }
        ptr++;
      }
      while (lptr < nnz_) {
        ret.Idx(ptr) = Idx(lptr);
        ret.Data(ptr) = alpha * Data(lptr);
        lptr++;
        ptr++;
      }
      while (rptr < rhs.nnz_) {
        ret.Idx(ptr) = rhs.Idx(rptr);
        ret.Data(ptr) = beta * rhs.Data(rptr);
        rptr++;
        ptr++;
      }
    } else {
      // Implement sparse dense linear combination
    }
  }

public:
  Vector() = default;
  explicit Vector(uint n) : dim_(n) {}
  explicit Vector(uint n, uint nnz) : dim_(n), nnz_(nnz) {}
  template <VectorType RHSType>
  Vector(const Vector<Sparse, RHSType> &A) : dim_(A.dim_), nnz_(A.nnz_) {}
  uint Idx(uint i) const { return storage_.InnerIdx(i); }
  uint &Idx(uint i) { return storage_.InnerIdx(i); }
  Real Data(uint i) const { return storage_.Data(i); }
  Real &Data(uint i) { return storage_.Data(i); }
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
  Real dot(const Vector &rhs) const {
    assert(dim_ == rhs.dim_);
    Real sum = 0.0;
    uint lptr = 0, rptr = 0;
    while (lptr < nnz_ && rptr < rhs.nnz_) {
      if (Idx(lptr) < rhs.Idx(rptr))
        lptr++;
      else if (Idx(lptr) > rhs.Idx(rptr))
        rptr++;
      else {
        sum += Data(lptr) * rhs.Data(rptr);
        lptr++;
        rptr++;
      }
    }
    return sum;
  }
  template <StorageType VecStorage, VectorType RHSType>
  void Saxpy(const Vector<VecStorage, RHSType> &rhs, Real k) {
    LinearUpdate(1, k, rhs);
  }
  template <StorageType VecStorage, VectorType RHSType>
  void ScaleAdd(const Vector<VecStorage, RHSType> &rhs, Real k) {
    LinearUpdate(k, 1, rhs);
  }
  template <StorageType VecStorage, VectorType RHSType>
  Vector<VecStorage, VecType> operator+(const Vector<VecStorage, RHSType> &rhs) const {
    return LinearCombine(1, 1, rhs);
  }
  Vector operator-() {
    Vector ret(*this);
    for (int i = 0; i < nnz_; i++)
      ret.Data(i) = -ret.Data(i);
    return ret;
  }
  Vector operator-(const Vector &rhs) const {
    return LinearCombine(1, -1, rhs);
  }
  Vector operator/(Real val) const {
    if (val == 0) {
      std::printf("Division by zero!");
      exit(-1);
    }
    Vector ret(dim_, nnz_);
    for (int i = 0; i < nnz_; i++)
      ret.Data(i) = Data(i) * val;
    return ret;
  }
  Vector operator*(Real val) const {
    Vector ret(dim_, nnz_);
    for (int i = 0; i < nnz_; i++)
      ret.Data(i) * val;
    return ret;
  }
  friend Vector operator*(Real val, const Vector &A) { return A * val; }
  Vector &inv() {
    for (int i = 0; i < nnz_; i++)
      Data(i) = 1.0 / Data(i);
    return *this;
  }
  Vector inv() const {
    Vector ret(*this);
    for (int i = 0; i < nnz_; i++)
      ret.Data(i) = 1.0 / Data(i);
    return ret;
  }
  Vector &operator+=(const Vector &rhs) {
    LinearUpdate(1, 1, rhs);
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
      Data(i) /= val;
    return *this;
  }
  template<VectorType RHSType>
  Vector &operator-=(const Vector<Sparse, RHSType> &rhs) {
    LinearUpdate(1, -1, rhs);
    return *this;
  }
  Vector &operator*=(Real val) {
    for (int i = 0; i < nnz_; i++)
      Data(i) *= val;
    return *this;
  }
  uint Dim() const { return dim_; }
  uint NonZeros() const { return nnz_; }
  Real RobustL2Norm() const {
    assert(dim_ > 0);
    if (!nnz_)
      return 0.0;
    Real maxv = std::abs(Data(0)), sum = 0.0;
    for (uint i = 1; i < nnz_; i++)
      maxv = std::max(std::abs(Data(i)), maxv);
    for (uint i = 0; i < nnz_; i++)
      sum += (Data(i) / maxv);
    return std::sqrt(sum) * maxv;
  }
  Real L2NormSqr() const {
    return dot(*this);
  }
  Real L2Norm() const {
    return std::sqrt(dot(*this));
  }
  Real RobustL2NormSqr() const {
    assert(dim_ > 0);
    if (!nnz_)
      return 0.0;
    Real maxv = std::abs(Data(0)), sum = 0.0;
    for (uint i = 1; i < dim_; i++)
      maxv = std::max(std::abs(Data(i)), maxv);
    for (uint i = 0; i < dim_; i++)
      sum += (Data(i) / maxv);
    return sum * maxv * maxv;
  }
  Real L1Norm() const {
    assert(dim_ > 0);
    Real sum = 0.0;
    for (uint i = 0; i < nnz_; i++)
      sum += std::abs(Data(i));
    return sum;
  }
  Real MaxNorm() const {
    Real maxv = 0.0;
    for(uint i = 0; i < nnz_; i++)
      maxv = std::max(maxv, std::abs(Data(i)));
    return maxv;
  }
  Vector Normalized() const {
    Real norm = L2Norm();
    Vector ret(dim_);
    for (int i = 0; i < nnz_; i++)
      Data(i) / norm;
    return ret;
  }
  friend std::ostream &operator<<(std::ostream &o, const Vector &A) {
    if (A.nnz_)
      o << "(" << A.Idx(0) << ", " << A.Data(0) << ")";
    for (int i = 1; i < A.nnz_; i++)
      o << std::endl << "(" << A.Idx(i) << ", " << A.Data(i) << ")";
    return o;
  }
};

} // namespace spmx

#endif // SPMX_SPMX_VECTOR_H
