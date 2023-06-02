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
    int *anc = new int[num_nodes_];
    for (uint i = 1; i < num_nodes_; i++) {
      for (uint j = A.OuterIndex(i);
           j < A.OuterIndex(i + 1) && A.InnerIdx(j) < i; j++) {
        uint x = A.InnerIdx(j);
        while(fa_[x] >= 0) {
          uint t = anc[x];
          anc[x] = static_cast<int>(i);
          x = t;
        }
        fa_[x] = anc[x] = static_cast<int>(i);
      }
    }
    delete[] anc;
  }
};
// TODO: Postorder

}



#endif // SPMX_ELIMINATION_TREE_H
