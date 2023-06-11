//
// Created by creeper on 23-3-13.
//

#ifndef SPMX_SPMX_UTILS_H
#define SPMX_SPMX_UTILS_H
#include <algorithm>
#include <climits>
#include <random>
#include <spmx-types.h>
#include <type_traits>

namespace spmx {
template <typename T> inline bool SimZero(T x) {
  if constexpr (std::is_same_v<T, double>)
    return x >= -DOUBLE_EPS && x <= DOUBLE_EPS;
  else if constexpr (std::is_same_v<T, float>)
    return x >= -FLOAT_EPS && x <= FLOAT_EPS;
  else
    return x == static_cast<T>(0);
}

template <typename T> inline bool Similar(T x, T y) {
  if constexpr (std::is_same_v<T, double>)
    return x - y >= -DOUBLE_EPS && x - y <= DOUBLE_EPS;
  else if constexpr (std::is_same_v<T, float>)
    return x - y >= -FLOAT_EPS && x - y <= FLOAT_EPS;
  else
    return x == y;
}

template <typename T> inline bool IsZero(T x) {
  static_assert(std::is_arithmetic_v<T>, "Must be arithmetic type");
  return x == static_cast<T>(0);
}

inline Real RandReal() {
  static std::random_device rd;
  static std::default_random_engine e(rd());
  static std::uniform_real_distribution<Real> distrib(0, 1);
  return distrib(e);
}
inline uint Randu() {
  static std::random_device rd;
  static std::default_random_engine e(rd());
  static std::uniform_int_distribution<> distrib(0, INT_MAX);
  return distrib(e);
}

template <StorageMajor major> bool TripletCmp(Triplet ta, Triplet tb) {
  if constexpr (major == RowMajor || major == Symmetric) {
    return std::get<0>(ta) == std::get<0>(tb)
               ? std::get<1>(ta) < std::get<1>(tb)
               : std::get<0>(ta) < std::get<0>(tb);
  } else
    return std::get<1>(ta) == std::get<1>(tb)
               ? std::get<0>(ta) < std::get<0>(tb)
               : std::get<1>(ta) < std::get<1>(tb);
}

template <typename Vec> inline Real L2NormSqr(const Vec &v) {
  Real sum = 0;
  if (traits<Vec>::storage == Sparse) {
    for (typename Vec::NonZeroIterator it(v); it(); ++it)
      sum += it.value() * it.value();
  } else {
    for (uint i = 0; i < v.Dim(); i++)
      sum += v(i) * v(i);
  }
  return sum;
}

template <typename Vec> inline Real L2Norm(const Vec &v) {
  Real sum = 0;
  if (traits<Vec>::storage == Sparse) {
    for (typename Vec::NonZeroIterator it(v); it(); ++it)
      sum += it.value() * it.value();
  } else {
    for (uint i = 0; i < v.Dim(); i++)
      sum += v(i) * v(i);
  }
  return std::sqrt(sum);
}

template <typename Vec> inline Real L1Norm(const Vec &v) {
  Real sum = 0;
  if (traits<Vec>::storage == Sparse) {
    for (typename Vec::NonZeroIterator it(v); it(); ++it)
      sum += std::abs(it.value());
  } else {
    for (uint i = 0; i < v.Dim(); i++)
      sum += std::abs(v(i));
  }
  return sum;
}

template <typename Lhs, typename Rhs> inline Real Dot(const Lhs& lhs, const Rhs& rhs) {

}

} // namespace spmx
#endif // SPMX_SPMX_UTILS_H
