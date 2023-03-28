//
// Created by creeper on 23-2-23.
//

#ifndef SPMX_TYPES_H
#define SPMX_TYPES_H
#include <tuple>
namespace SpmX
{
    using Real = double;
    using uint = unsigned int;
    using Triplet = std::tuple<uint, uint, Real>;
    const Real EPS = 1e-15;
}
#endif //SPMX_TYPES_H
