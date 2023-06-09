//
// Created by creeper on 23-5-17.
//

#ifndef SPMX_ELIMINATION_TREE_H
#define SPMX_ELIMINATION_TREE_H

#include <queue>
#include <sparse-matrix.h>
#include <vector>

namespace spmx {

class EliminationTree {
private:
  int *fa_ = nullptr;
  uint num_nodes_ = 0;
public:
  EliminationTree() = default;
  explicit EliminationTree(uint n) : num_nodes_(n) {
    fa_ = new int[n];
    memset(fa_, -1, sizeof(uint) * n);
  }
  template<typename MatDerived>
  void BuildFromMatrix(const SparseMatrixBase<MatDerived> &A) {

  }
};

}



#endif // SPMX_ELIMINATION_TREE_H
