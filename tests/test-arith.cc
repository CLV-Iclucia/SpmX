#include <ctime>
#include <fstream>
#include <sparse-matrix.h>
#include <spmx-utils.h>

const uint TEST_SET = 1u << 0;
const uint TEST_ADD = 1u << 1;
const uint TEST_SUB = 1u << 2;
const uint TEST_MV_MUL = 1u << 3;

using namespace spmx;
const uint MAX_ROWS = 600, MAX_COLS = 800, MAX_NNZ = 200;
const uint TESTS = TEST_ADD;
Triplet tList[MAX_NNZ];
const int MAX_CASES = 100;
static Real golden[MAX_ROWS][MAX_COLS];

static Real A[MAX_ROWS][MAX_COLS], B[MAX_ROWS][MAX_COLS];
static Real v[MAX_COLS], golden_v[MAX_ROWS];
void RandFillMat(Real mat[][800], uint m, uint n, uint nnz) {
  for (int i = 0; i < nnz; i++) {
    uint x = Randu();
    uint y = Randu();
    Real val = Randu() * RandReal();
    tList[i] = {x, y, val};
    mat[x][y] += val;
  }
}

bool TestSame(Real stdmat[][800], const SparseMatrixXd &spm) {
  static std::vector<Triplet> v;
  v.clear();
  spm.toTriplets(v);
  v.reserve(spm.NonZeroEst());
  uint nnz = 0;
  for (uint i = 0; i < spm.Rows(); i++)
    for (uint j = 0; j < spm.Cols(); j++)
      if (!iszero(stdmat[i][j]))
        nnz++;
  if (nnz != spm.NonZeroEst()) {
    std::cerr << "Testing same: wrong non-zeros" << std::endl;
    std::cerr << "Expected non-zeros: " << nnz << std::endl
              << "Your non-zeros: " << spm.NonZeroEst() << std::endl;
    return false;
  }
  for (uint i = 0; i < spm.NonZeroEst(); i++)
    if (!Similar(std::get<2>(v[i]), stdmat[std::get<0>(v[i])][std::get<1>(v[i])])) {
      std::cerr << "Testing same: wrong value" << std::endl;
      return false;
    }
  return true;
}

bool TestSame(Real stdv[], const Vector<Dense> &V) {
  for (uint i = 0; i < V.Dim(); i++) {
    if (!Similar(stdv[i], V[i])) {
      std::cerr << "Testing same: wrong value" << std::endl;
      return false;
    }
  }
  return true;
}

void WriteWrongCase(uint m, uint n, uint nnz) {
  std::cerr << m << " " << n << " " << nnz << std::endl;
  for (uint i = 0; i < nnz; i++)
    std::cerr << std::get<0>(tList[i]) << " " << std::get<1>(tList[i]) << " "
              << std::get<2>(tList[i]) << std::endl;
}

void TestSet() {
  uint kase = 0;
  printf("Running tests on setFromTriplets...\n");
  while (kase < MAX_CASES) {
    memset(golden, 0, sizeof(golden));
    uint m = Randu() % MAX_ROWS + 1;
    uint n = Randu() % MAX_COLS + 1;
    uint nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(golden, m, n, nnz);
    SparseMatrixXd spm;
    spm.Resize(m, n);
    spm.SetFromTriplets(tList, tList + nnz);
    if (!TestSame(golden, spm)) {
      std::cerr << "Failed test setFromTriplets. Failing case is:" << std::endl;
      std::cerr << "A:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(A[i][j]))
            std::cerr << i << " " << j << " " << A[i][j] << std::endl;
      std::cerr << "B:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(B[i][j]))
            std::cerr << i << " " << j << " " << B[i][j] << std::endl;
      std::cerr << "std:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(golden[i][j]))
            std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
      std::cerr << "Your result is" << std::endl;
      std::cerr << spm << std::endl;
      exit(-1);
    }
    printf("Passed test case %d\n", ++kase);
  }
  printf("Passed all tests on setFromTriplets.\n");
}

void test_add() {
  uint kase = 0;
  printf("Running tests on add...\n");
  while (kase < MAX_CASES) {
    memset(golden, 0, sizeof(golden));
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    uint m = Randu() % MAX_ROWS + 1;
    uint n = Randu() % MAX_COLS + 1;
    uint nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(A, m, n, nnz);
    SparseMatrixXd spmA, spmB, spm;
    spmA.Resize(m, n);
    spmA.SetFromTriplets(tList, tList + nnz);
    nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(B, m, n, nnz);
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        golden[i][j] = A[i][j] + B[i][j];
    spmB.Resize(m, n);
    spmB.SetFromTriplets(tList, tList + nnz);
    spm = spmA + spmB;
    if (!TestSame(golden, spm)) {
      std::cerr << "Failed test add. Failing case is:" << std::endl;
      std::cerr << "A:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(A[i][j]))
            std::cerr << i << " " << j << " " << A[i][j] << std::endl;
      std::cerr << "B:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(B[i][j]))
            std::cerr << i << " " << j << " " << B[i][j] << std::endl;
      std::cerr << "std:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(golden[i][j]))
            std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
      std::cerr << "Your result is" << std::endl;
      std::cerr << "spm:" << std::endl << spm << std::endl;
      exit(-1);
    }
    printf("Passed test case %d\n", ++kase);
  }
  printf("Passed all tests on add.\n");
}

void test_sub() {
  uint kase = 0;
  printf("Running tests on sub...\n");
  while (kase < MAX_CASES) {
    memset(golden, 0, sizeof(golden));
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    uint m = Randu() % MAX_ROWS + 1;
    uint n = Randu() % MAX_COLS + 1;
    uint nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(A, m, n, nnz);
    SparseMatrixXd spmA, spmB, spm;
    spmA.Resize(m, n);
    spmA.SetFromTriplets(tList, tList + nnz);
    nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(B, m, n, nnz);
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        golden[i][j] = A[i][j] - B[i][j];
    spmB.Resize(m, n);
    spmB.SetFromTriplets(tList, tList + nnz);
    spm = spmA - spmB;
    if (!TestSame(golden, spm)) {
      std::cerr << "Failed test sub. Failing case is:" << std::endl;
      std::cerr << "A:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(A[i][j]))
            std::cerr << i << " " << j << " " << A[i][j] << std::endl;
      std::cerr << "B:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(B[i][j]))
            std::cerr << i << " " << j << " " << B[i][j] << std::endl;
      std::cerr << "std:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(golden[i][j]))
            std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
      std::cerr << "Your result is" << std::endl;
      std::cerr << "spm:" << std::endl << spm << std::endl;
      exit(-1);
    }
    printf("Passed test case %d\n", ++kase);
  }
  printf("Passed all tests on sub.\n");
}

void test_mv_mul() {
  uint kase = 0;
  printf("Running tests on mat-vec multiplication...\n");
  while (kase < MAX_CASES) {
    memset(golden_v, 0, sizeof(golden_v));
    memset(A, 0, sizeof(A));
    memset(v, 0, sizeof(v));
    uint m = Randu() % MAX_ROWS + 1;
    uint n = Randu() % MAX_COLS + 1;
    uint nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(A, m, n, nnz);
    SparseMatrixXd spm;
    Vector calc_v(n);
    spm.Resize(m, n);
    spm.SetFromTriplets(tList, tList + nnz);
    for (uint i = 0; i < n; i++) {
    }
    if (!TestSame(golden_v, calc_v)) {
      std::cerr << "Failed test mat-vec-multiplication. Failing case is:"
                << std::endl;
      std::cerr << "A:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(A[i][j]))
            std::cerr << i << " " << j << " " << A[i][j] << std::endl;
      std::cerr << "B:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(B[i][j]))
            std::cerr << i << " " << j << " " << B[i][j] << std::endl;
      std::cerr << "std:" << std::endl;
      for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
          if (!iszero(golden[i][j]))
            std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
      std::cerr << "Your result is" << std::endl;
      std::cerr << "spm:" << std::endl << spm << std::endl;
      exit(-1);
    }
    printf("Passed test case %d\n", ++kase);
  }
  printf("Passed all tests on mat-vec multiplication.\n");
}

int main(int argc, char **argv) {
  printf("Start testing arithmetics...\n");
  if constexpr (TESTS & TEST_SET)
    TestSet();
  if constexpr (TESTS & TEST_ADD)
    test_add();
  if constexpr (TESTS & TEST_SUB)
    test_sub();
  printf("All tests passed!\n");
}
