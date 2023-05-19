//
// Created by creeper on 23-5-17.
//

#ifndef SPMX_ELIMINATION_TREE_H
#define SPMX_ELIMINATION_TREE_H

#include <queue>
#include <sparse-matrix.h>
#include <vector>

class EliminationTree {
private:
  uint *fa_ = nullptr;
  uint *compressed_childs_ = nullptr;
  uint num_nodes_ = 0;
public:
  EliminationTree() = default;
  explicit EliminationTree(uint n) : num_nodes_(n) {
    fa_ = new uint[n];
    memset(fa_, 0, sizeof(uint) * n);
  }
  void BuildFromMatrix(const SparseMatrix &A) {
    uint *anc = new uint[num_nodes_];
    for (uint i = 1; i < num_nodes_; i++) {
      for (uint j = A.OuterIndex(i);
           j < A.OuterIndex(i + 1) && A.InnerIndex(j) < i; j++) {
        uint x = A.InnerIndex(j);
        while(!fa_[x])
          x = anc[x];
        fa_[x] = anc[x] = i;
        x = A.InnerIndex(j);
        while(x != i) {
          uint t = anc[x];
          anc[x] = i;
          x = anc[x];
        }
      }
    }
    delete[] anc;
  }
};

#endif // SPMX_ELIMINATION_TREE_H
