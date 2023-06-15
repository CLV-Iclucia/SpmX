//
// Created by creeper on 23-6-13.
//
//
// Created by creeper on 23-6-7.
//
#include <expressions.h>
#include <iterative-solvers.h>
#include <sparse-cholesky/simplicial-cholesky.h>
using namespace spmx;

const uint TEST_CHOLESKY = 1u << 0;
const uint TEST_CG = 1u << 1;

using namespace spmx;
const uint MAX_SZ = 1 << 17;
const uint TESTS = TEST_CG;
Triplet tList[MAX_SZ * 7];
const int MAX_CASES = 1;

void RandFillMat(uint &nnz) {
  nnz = 0;
  for (int i = 0; i < MAX_SZ; i++) {
    for (int j = -3; j <= 3; j++) {
      if (i + j >= 0 && i + j < MAX_SZ) {
        if (j == 0)
          tList[nnz++] = {i, i, 1.0};
        else
          tList[nnz++] = {i, i + j, -1.0 / 6.0};
      }
    }
  }
}

template <uint Test> void TestSolve() {
  uint kase = 0;
  while (kase < MAX_CASES) {
    uint nnz;
    std::cout << MAX_SZ << std::endl;
    RandFillMat(nnz);
    std::cout << MAX_SZ << std::endl;
    SparseMatrix<0, 0, Sparse, Symmetric> mat(MAX_SZ, MAX_SZ);
    std::cout << MAX_SZ << std::endl;
    mat.SetFromTriplets(tList, tList + nnz);
    Vector<Dense> b(MAX_SZ);
    for (uint i = 0; i < MAX_SZ; i++)
      b(i) = 1.0;
    Vector<Dense> ans;
    if constexpr (Test == TEST_CHOLESKY) {
      SimplicialLLT<SparseMatrix<0, 0, Sparse, Symmetric>> solver;
      solver.Compute(mat);
      if (solver.info() != Success) {
        std::printf("Computing failed.\n");
        break;
      }
      ans = solver.Solve(b);
    }
    std::cout << "Start benchmarking conjugate gradient solver with 7-points "
                 "Poisson matrix."
              << std::endl;
    time_t begin = time(nullptr);
    if constexpr (Test == TEST_CG) {
      ConjugateGradientSolver<SparseMatrix<0, 0, Sparse, Symmetric>> solver;
      solver.SetPrecision(1e-2);
      ans = solver.Solve(mat, b);
      std::cout << "case " << kase << ". " << solver.Rounds() << std::endl;
    }
    time_t end = time(nullptr);
    std::cout << end - begin << std::endl;
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
}