//
// Created by creeper on 23-6-7.
//
#include <sparse-cholesky/simplicial-llt.h>
using namespace spmx;

int main() {
  SparseMatrix<100, 100, Sparse, RowMajor> mat;
  SimplicialLLT<SparseMatrix<100, 100, Sparse, RowMajor>> solver;
  int a = traits<SparseMatrix<100, 100, Sparse, RowMajor>>::nCols;
  std::cout << a << std::endl;
}