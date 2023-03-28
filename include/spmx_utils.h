//
// Created by creeper on 23-3-13.
//

#ifndef SPMX_SPMX_UTILS_H
#define SPMX_SPMX_UTILS_H
#include <spmx_types.h>
namespace SpmX
{
    inline bool isZero(Real x)
    {
        return x >= -EPS && x <= EPS;
    }
}
#endif //SPMX_SPMX_UTILS_H
