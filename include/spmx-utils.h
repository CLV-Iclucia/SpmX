//
// Created by creeper on 23-3-13.
//

#ifndef SPMX_SPMX_UTILS_H
#define SPMX_SPMX_UTILS_H
#include <spmx-types.h>
#include <climits>
#include <random>
#include <type_traits>
#include <algorithm>

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

template <StorageMajor major>
bool TripletCmp(Triplet ta, Triplet tb) {
  if constexpr (major == RowMajor) {
    return std::get<0>(ta) < std::get<0>(tb);
  } else return std::get<1>(ta) < std::get<1>(tb);
}

template<typename Iterator, StorageMajor major>
inline void SortByMajor(Iterator begin, Iterator end) {

}

} // namespace spmx
#endif // SPMX_SPMX_UTILS_H
