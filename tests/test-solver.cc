//
// Created by creeper on 23-6-7.
//
#include <expressions.h>
#include <sparse-cholesky/simplicial-cholesky.h>
#include <iterative-solvers.h>
using namespace spmx;

const uint TEST_CHOLESKY = 1u << 0;
const uint TEST_CG = 1u << 1;

using namespace spmx;
const uint MAX_SZ = 100000, MAX_UPPER = 2000000;
const uint TESTS = TEST_CG;
Triplet tList[MAX_UPPER << 1];
const int MAX_CASES = 1;
static int res[MAX_CASES];

void RandFillMat(uint n, uint& nnz) {
  nnz = 0;
  uint upper = Randu() % MAX_SZ + 1;
  for (int i = 0; i < n; i++) {
    tList[nnz++] = {i, i, RandReal()};
  }
  for (uint i = 0; i < upper; i++) {
    uint x = Randu() % n;
    uint y = Randu() % n;
    Real val = RandReal();
    tList[nnz++] = {x, y, val};
    if(x != y) {
      tList[nnz++] = {y, x, val};
    }
  }
}

template <uint Test>
void TestSolve() {
  uint kase = 0;
  while (kase < MAX_CASES) {
    uint n = Randu() % MAX_SZ + 1;
    uint nnz;
    RandFillMat(n, nnz);
    SparseMatrix<0, 0, Sparse, Symmetric> mat(n, n);
    mat.SetFromTriplets(tList, tList + nnz);
    Vector<Dense> b(n);
    for(uint i = 0; i < n; i++)
      b(i) = RandReal();
    Vector<Dense> ans;
    std::cout << nnz << std::endl;
    if constexpr (Test == TEST_CHOLESKY) {
      SimplicialLLT<SparseMatrix<0, 0, Sparse, Symmetric>> solver;
      solver.Compute(mat);
      if (solver.info() != Success) {
        std::printf("Computing failed.\n");
        break;
      }
      ans = solver.Solve(b);
    }
    if constexpr (Test == TEST_CG) {
      ConjugateGradientSolver<SparseMatrix<0, 0, Sparse, Symmetric>> solver;
      ans = solver.Solve(mat, b);
      std::cout << "case " << kase << ". " << solver.Rounds() << std::endl;
    }
    Real eps = L2Norm(mat * ans - b);
    if (!SimZero(eps)) {
      std::cerr << "Warning: the error is " << eps << std::endl;
    }
    kase++;
  }
}

int main() {
  if constexpr (TESTS & TEST_CG)
    TestSolve<TEST_CG>();
  if constexpr (TESTS & TEST_CHOLESKY)
    TestSolve<TEST_CHOLESKY>();
}