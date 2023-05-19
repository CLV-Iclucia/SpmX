//
// Created by creeper on 23-5-15.
//

#ifndef SPMX_MY_TINY_BLAS_H
#define SPMX_MY_TINY_BLAS_H

#include <spmx-types.h>
#include <spmx-utils.h>
#include <thread>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#ifdef SIMD_SSE2_ENABLED
#include <immintrin.h>
#endif

namespace spmx {

inline void ds_saxpy(Real *__restrict__ a, const Real *__restrict__ b, Real k,
                     uint n) {
  for (int i = 0; i < n; i += 2) {
    __m128d b_vec = _mm_load_pd(b + i);
    __m128d a_vec = _mm_load_pd(a + i);
    __m128d k_vec = _mm_set1_pd(k);
    __m128d kb_vec = _mm_mul_pd(b_vec, k_vec);
    __m128d r_vec = _mm_add_pd(a_vec, kb_vec);
    _mm_store_pd(a + i, r_vec);
  }
}

/**
 * a kernel for matrix a of size (6, 16)
 * @param a
 * @param b
 * @param c
 */
inline void ds_gemm6x16(const Real *__restrict__ a, const Real *__restrict__ b,
                         Real *__restrict__ c) {
  
}

inline void ds_gemm(const Real *__restrict__ a, const Real *__restrict__ b,
                    Real *__restrict__ c, uint m, uint p, uint n) {
#define A(i, j) a[p * i + j]
#define B(i, j) b[n * i + j]
#define C(i, j) c[n * i + j]

  memset(c, 0, sizeof(Real) * m * n);
  for(uint i = 0; i < m; i++)
    for(uint j = 0; j < n; j++)


#undef A
#undef B
#undef C
}

/**
 * compute dot(a[0:n], b[0:n])
 */
inline Real ds_dot(const Real *__restrict__ a, const Real *__restrict__ b,
                   uint n) {
  Real sum = 0.0;

  for (int i = 0; i < n; i += 2) {
    __m128d b_vec = _mm_load_pd(b + i);
    __m128d a_vec = _mm_load_pd(a + i);
    __m128d s_vec = _mm_mul_pd(a_vec, b_vec);
    sum += s_vec[0] + s_vec[1];
  }
  return sum;
}
inline void ds_add(const Real *__restrict__ a, const Real *__restrict__ b,
                   Real *__restrict__ c, uint n) {
  Real sum = 0.0;
  for (int i = 0; i < n; i++) {
    __m128d b_vec = _mm_load_pd(b + i);
    __m128d a_vec = _mm_load_pd(a + i);
    __m128d r_vec = _mm_add_pd(a_vec, b_vec);
    _mm_store_pd(c + i, r_vec);
  }
}

inline void ds_sp_add(const Real *__restrict__ a, const Real *__restrict__ idx,
                      const Real *__restrict__ val, Real *__restrict__ ret,
                      uint n, uint nnz) {}

inline Real ds_self_dot(Real *__restrict__ a, uint n) {
  Real sum = 0.0;
  for (int i = 0; i < n; i++) {
    __m128d a_vec = _mm_load_pd(a + i);
    __m128d r_vec = _mm_mul_pd(a_vec, a_vec);
    sum += r_vec[0] + r_vec[1];
  }
  return sum;
}

inline void gather(const uint *__restrict__ idx, const uint *__restrict__ val,
                   Real *__restrict__ a, uint nnz) {
  for (uint i = 0; i < nnz; i++)
    a[idx[i]] = val[i];
}

inline Real ds_sp_dot(const Real *__restrict__ a, const uint *__restrict__ idx,
                      const Real *__restrict__ val, uint n, uint nnz) {}

inline void scatter(Real *__restrict__ a, uint *__restrict__ idx,
                    uint *__restrict__ val, uint n) {
  uint cnt = 0;
  for (uint i = 0; i < n; i++) {
    if (!IsZero(a[i])) {
      idx[cnt] = a[i];
      val[cnt++] = i;
    }
  }
}

inline Real sp_dot(uint *__restrict__ idx_a, uint *__restrict__ idx_b,
                   Real *__restrict__ val_a, Real *__restrict__ val_b, uint n) {
}

inline Real sp_ds_dot(const uint *__restrict__ idx_a,
                      const Real *__restrict__ val_a,
                      const Real *__restrict__ b, uint nnz, uint n) {
  Real sum = 0.0;
  for (uint i = 0; i < nnz; i++)
    sum += val_a[i] * b[idx_a[i]];
  return sum;
}

inline void sp_ds_gemv(uint *__restrict__ r_ptr, uint *__restrict__ c_idx,
                       Real *__restrict__ m_data, Real *__restrict__ v, uint m,
                       uint n, uint nnz) {
  for (uint i = 0; i < m; i++) {

  }
}

} // namespace spmx

#endif // SPMX_MY_TINY_BLAS_H
