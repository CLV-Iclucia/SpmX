//
// Created by creeper on 23-6-10.
//
#include <expressions.h>
#include <fstream>
#include <sparse-matrix.h>
#include <spmx-utils.h>

const uint TEST_LIN = 1u << 0;
const uint TEST_SET = 1u << 1;
const uint TEST_MV_MUL = 1u << 2;

using namespace spmx;
const uint MAX_ROWS = 100000, MAX_COLS = 100000, MAX_NNZ = 1000000;
const uint TESTS = TEST_LIN | TEST_SET;
Triplet tList[MAX_NNZ];
const int MAX_CASES = 100;
static int res[MAX_CASES];
void RandFillMat(uint m, uint n, uint nnz) {
  for (uint i = 0; i < nnz; i++) {
    uint x = Randu() % m;
    uint y = Randu() % n;
    Real val = RandReal();
    tList[i] = {x, y, val};
  }
}

void BenchLin() {
  uint kase = 0;
  printf("Running tests on linear expressions...\n");
  while (kase < MAX_CASES) {
    uint m = Randu() % MAX_ROWS + 1;
    uint n = Randu() % MAX_COLS + 1;
    uint nnz = Randu() % MAX_NNZ + 1;
    Real a = RandReal();
    Real b = RandReal();
    RandFillMat(m, n, nnz);
    SparseMatrixXd spmA, spmB, spm;
    spmA.Resize(m, n);
    spmA.SetFromTriplets(tList, tList + nnz);
    nnz = Randu() % MAX_NNZ + 1;
    RandFillMat(m, n, nnz);
    spmB.Resize(m, n);
    spmB.SetFromTriplets(tList, tList + nnz);
    spm = a * spmA + b * spmB;
    res[kase++] = spm.NonZeroEst();
  }
}

int main(int argc, char **argv) {
  BenchLin();
}