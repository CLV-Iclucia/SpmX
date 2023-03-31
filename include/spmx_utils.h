//
// Created by creeper on 23-3-13.
//

#ifndef SPMX_SPMX_UTILS_H
#define SPMX_SPMX_UTILS_H
#include <spmx_types.h>
namespace SpmX
{
    template<typename T>
    inline bool isZero(T x)
    {
        if constexpr (std::is_same_v<T, double>)
            return x >= -DOUBLE_EPS && x <= DOUBLE_EPS;
        else if constexpr (std::is_same_v<T, float>)
            return x >= -FLOAT_EPS && x <= FLOAT_EPS;
        else return x == static_cast<T>(0);
    }

    template<typename T>
    inline bool isEqual(T x, T y)
    {
        if constexpr (std::is_same_v<T, double>)
            return x - y >= -DOUBLE_EPS && x - y <= DOUBLE_EPS;
        else if constexpr (std::is_same_v<T, float>)
            return x - y >= -FLOAT_EPS && x - y <= FLOAT_EPS;
        else return x == y;
    }
}
#endif //SPMX_SPMX_UTILS_H
