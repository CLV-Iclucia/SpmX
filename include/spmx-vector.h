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


} // namespace spmx

#endif // SPMX_SPMX_VECTOR_H
