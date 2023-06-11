//
// Created by creeper on 23-6-7.
//
#include <sparse-cholesky/simplicial-llt.h>
using namespace spmx;

const uint TEST_LIN = 1u << 0;
const uint TEST_SET = 1u << 1;
const uint TEST_MV_MUL = 1u << 2;

using namespace spmx;
const uint MAX_SZ = 100, MAX_UPPER = 1000;
const uint TESTS = TEST_LIN | TEST_SET;
Triplet tList[MAX_UPPER << 1];
Real A[MAX_SZ][MAX_SZ];
const int MAX_CASES = 100;
static int res[MAX_CASES];

void RandFillMat(uint n, uint& nnz) {
  nnz = 0;
  uint upper = Randu() % MAX_SZ + 1;
  for (uint i = 0; i < upper; i++) {
    uint x = Randu() % n;
    uint y = Randu() % n;
    Real val = RandReal();
    A[x][y] = val;
    tList[nnz++] = {x, y, val};
    if(x != y) {
      tList[nnz++] = {y, x, val};
      A[y][x] = val;
    }
  }
}

void TestSolve() {
  uint kase = 0;
  while (kase < MAX_CASES) {
    uint n = Randu() % MAX_SZ + 1;
    uint nnz;
    RandFillMat(n, nnz);
    SparseMatrix<0, 0, Sparse, Symmetric> mat(n, n);
    mat.SetFromTriplets(tList, tList + nnz);
    SimplicialLLT<SparseMatrix<0, 0, Sparse, Symmetric>> solver;
    solver.Compute(mat);
    if (solver.info() != Success) {
      std::printf("Computing failed.\n");
      continue;
    }
    Vector<Dense> b(n);
    for(uint i = 0; i < n; i++)
      b(i) = RandReal();
    Vector<Dense> ans = solver.Solve(b);

    kase++;
  }
}

int main() {

  TestSolve();
}