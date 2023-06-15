//
// Created by creeper on 23-5-17.
//

#ifndef SPMX_PRECONDITIONERS_H
#define SPMX_PRECONDITIONERS_H
#include <spmx-types.h>
#include <spmx-utils.h>
#include <type-utils.h>
#include <sparse-matrix.h>
namespace spmx {

/**
 * A naive diagonal preconditioner.
 * @tparam MatType
 */
template <typename MatType> class DiagonalPreconditioner {
  static constexpr uint Size = traits<MatType>::nRows;
public:
  void Solve(const MatType &A, const Vector<Dense, Size> &b, Vector<Dense, Size> &y) {

  }
};

} // namespace spmx

#endif // SPMX_PRECONDITIONERS_H
