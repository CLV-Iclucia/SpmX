//
// Created by creeper on 23-4-8.
//
#include <SparseMatrix.h>
#include <random>
#include <ctime>
#include <spmx_utils.h>

const uint TEST_SET = 1u << 0;
const uint TEST_ADD = 1u << 1;
const uint TEST_SUB = 1u << 2;
const uint TEST_MV_MUL = 1u << 3;

using namespace SpmX;
const uint MAX_ROWS = 6000, MAX_COLS = 8000, MAX_NNZ = 20000;
const uint TESTS = TEST_ADD;
Triplet tList[MAX_NNZ];
const int MAX_CASES = 10, MAX_REPEATS = 1000;
void rand_fill_mat(uint m, uint n, uint nnz)
{
    for(int i = 0; i < nnz; i++)
    {
        uint x = rand() % m;
        uint y = rand() % n;
        Real val = 1.0 * rand() / (rand() + 1.0);
        tList[i] = {x, y, val};
    }
}

void test_set()
{
    uint kase = 0;
    printf("Running tests on set.\n");
    while(kase < MAX_CASES)
    {
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(m, n, nnz);
        DynamicSparseMatrix spm;
        spm.resize(m, n);
        spm.setFromTriplets(tList, tList + nnz);
        kase++;
    }
    printf("Tests on set done.\n");
}

void test_add()
{
    uint kase = 0;
    printf("Running tests on add.\n");
    while(kase < MAX_CASES)
    {
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(m, n, nnz);
        DynamicSparseMatrix spmA, spmB, spm;
        spmA.resize(m, n);
        spmA.setFromTriplets(tList, tList + nnz);
        nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(m, n, nnz);
        spmB.resize(m, n);
        spmB.setFromTriplets(tList, tList + nnz);
#pragma optimize("", off)
        for(uint i = 0; i < MAX_REPEATS; i++)
        {
            spm = spmA + spmB;
            spm = spmA - spmB;
        }
#pragma optimize("", on)
        kase++;
    }
    printf("Tests on add done.\n");
}

void test_sub()
{
    uint kase = 0;
    printf("Running tests on sub.\n");
    while(kase < MAX_CASES)
    {
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(m, n, nnz);
        DynamicSparseMatrix spmA, spmB, spm;
        spmA.resize(m, n);
        spmA.setFromTriplets(tList, tList + nnz);
        nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(m, n, nnz);
        spmB.resize(m, n);
        spmB.setFromTriplets(tList, tList + nnz);
        spm = spmA - spmB;
        kase++;
    }
    printf("Tests on sub done.\n");
}

void test_mv_mul()
{
    uint kase = 0;
    printf("Running tests on mat-vec multiplication.\n");
    while(kase < MAX_CASES)
    {
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(m, n, nnz);
        DynamicSparseMatrix spm;
        Vector calc_v(n);
        spm.resize(m, n);
        spm.setFromTriplets(tList, tList + nnz);
        for(uint i = 0; i < n; i++)
        {

        }
        kase++;
    }
    printf("Passed all tests on mat-vec multiplication.\n");
}

int main()
{
    srand(time(0));

    printf("Start benchmarking.\n");
    if constexpr (TESTS & TEST_SET)
        test_set();
    if constexpr (TESTS & TEST_ADD)
        test_add();
    if constexpr (TESTS & TEST_SUB)
        test_sub();
    printf("Benchmarking done.\n");
}