#include <SparseMatrix.h>
#include <random>
#include <ctime>
#include <spmx_utils.h>
#include <fstream>

const uint TEST_SET = 1u << 0;
const uint TEST_ADD = 1u << 1;
const uint TEST_SUB = 1u << 2;
const uint TEST_MV_MUL = 1u << 3;

using namespace SpmX;
const uint MAX_ROWS = 600, MAX_COLS = 800, MAX_NNZ = 200;
const uint TESTS = TEST_ADD;
Triplet tList[MAX_NNZ];
const int MAX_CASES = 100;
static Real golden[MAX_ROWS][MAX_COLS];

static Real A[MAX_ROWS][MAX_COLS], B[MAX_ROWS][MAX_COLS];
static Real v[MAX_COLS], golden_v[MAX_ROWS];
void rand_fill_mat(Real mat[][MAX_COLS], uint m, uint n, uint nnz)
{
    for(int i = 0; i < nnz; i++)
    {
        uint x = rand() % m;
        uint y = rand() % n;
        Real val = 1.0 * rand() / (rand() + 1.0);
        tList[i] = {x, y, val};
        mat[x][y] += val;
    }
}

bool test_same(Real stdmat[][MAX_COLS], const DynamicSparseMatrix& spm)
{
    static std::vector<Triplet> v;
    v.clear();
    spm.toTriplets(v);
    v.reserve(spm.nonZeros());
    uint nnz = 0;
    for(uint i = 0; i < spm.rows(); i++)
        for(uint j = 0; j < spm.cols(); j++)
            if(!isZero(stdmat[i][j])) nnz++;
    if(nnz != spm.nonZeros())
    {
        std::cerr << "Testing same: wrong non-zeros" << std::endl;
        std::cerr << "Expected non-zeros: " << nnz << std::endl << "Your non-zeros: " << spm.nonZeros() << std::endl;
        return false;
    }
    for(uint i = 0; i < spm.nonZeros(); i++)
        if(!isEqual(std::get<2>(v[i]), stdmat[std::get<0>(v[i])][std::get<1>(v[i])]))
        {
            std::cerr << "Testing same: wrong value" << std::endl;
            return false;
        }
    return true;
}

bool test_same(Real stdv[], const Vector& V)
{
    for(uint i = 0; i < V.dim(); i++)
    {
        if(!isEqual(stdv[i], V[i]))
        {
            std::cerr << "Testing same: wrong value" << std::endl;
            return false;
        }
    }
    return true;
}

void write_wrong_case(uint m, uint n, uint nnz)
{
    std::cerr << m << " " << n << " " << nnz << std::endl;
    for(uint i = 0; i < nnz; i++)
        std::cerr << std::get<0>(tList[i]) << " " << std::get<1>(tList[i]) << " " << std::get<2>(tList[i]) << std::endl;
}

void test_set()
{
    uint kase = 0;
    printf("Running tests on setFromTriplets...\n");
    while(kase < MAX_CASES)
    {
        memset(golden, 0, sizeof(golden));
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(golden, m, n, nnz);
        DynamicSparseMatrix spm;
        spm.resize(m, n);
        spm.setFromTriplets(tList, tList + nnz);
        if(!test_same(golden, spm))
        {
            std::cerr << "Failed test setFromTriplets. Failing case is:" << std::endl;
            std::cerr << "A:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(A[i][j])) std::cerr << i << " " << j << " " << A[i][j] << std::endl;
            std::cerr << "B:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(B[i][j])) std::cerr << i << " " << j << " " << B[i][j] << std::endl;
            std::cerr << "std:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(golden[i][j])) std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
            std::cerr << "Your result is" << std::endl;
            std::cerr << spm << std::endl;
            exit(-1);
        }
        printf("Passed test case %d\n", ++kase);
    }
    printf("Passed all tests on setFromTriplets.\n");
}

void test_add()
{
    uint kase = 0;
    printf("Running tests on add...\n");
    while(kase < MAX_CASES)
    {
        memset(golden, 0, sizeof(golden));
        memset(A, 0, sizeof(A));
        memset(B, 0, sizeof(B));
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(A, m, n, nnz);
        DynamicSparseMatrix spmA, spmB, spm;
        spmA.resize(m, n);
        spmA.setFromTriplets(tList, tList + nnz);
        nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(B, m, n, nnz);
        for(uint i = 0; i < m; i++)
            for(uint j = 0; j < n; j++)
                golden[i][j] = A[i][j] + B[i][j];
        spmB.resize(m, n);
        spmB.setFromTriplets(tList, tList + nnz);
        spm = spmA + spmB;
        if(!test_same(golden, spm))
        {
            std::cerr << "Failed test add. Failing case is:" << std::endl;
            std::cerr << "A:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(A[i][j])) std::cerr << i << " " << j << " " << A[i][j] << std::endl;
            std::cerr << "B:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(B[i][j])) std::cerr << i << " " << j << " " << B[i][j] << std::endl;
            std::cerr << "std:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(golden[i][j])) std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
            std::cerr << "Your result is" << std::endl;
            std::cerr << "spm:" << std::endl << spm << std::endl;
            exit(-1);
        }
        printf("Passed test case %d\n", ++kase);
    }
    printf("Passed all tests on add.\n");
}

void test_sub()
{
    uint kase = 0;
    printf("Running tests on sub...\n");
    while(kase < MAX_CASES)
    {
        memset(golden, 0, sizeof(golden));
        memset(A, 0, sizeof(A));
        memset(B, 0, sizeof(B));
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(A, m, n, nnz);
        DynamicSparseMatrix spmA, spmB, spm;
        spmA.resize(m, n);
        spmA.setFromTriplets(tList, tList + nnz);
        nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(B, m, n, nnz);
        for(uint i = 0; i < m; i++)
            for(uint j = 0; j < n; j++)
                golden[i][j] = A[i][j] - B[i][j];
        spmB.resize(m, n);
        spmB.setFromTriplets(tList, tList + nnz);
        spm = spmA - spmB;
        if(!test_same(golden, spm))
        {
            std::cerr << "Failed test sub. Failing case is:" << std::endl;
            std::cerr << "A:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(A[i][j])) std::cerr << i << " " << j << " " << A[i][j] << std::endl;
            std::cerr << "B:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(B[i][j])) std::cerr << i << " " << j << " " << B[i][j] << std::endl;
            std::cerr << "std:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(golden[i][j])) std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
            std::cerr << "Your result is" << std::endl;
            std::cerr << "spm:" << std::endl << spm << std::endl;
            exit(-1);
        }
        printf("Passed test case %d\n", ++kase);
    }
    printf("Passed all tests on sub.\n");
}

void test_mv_mul()
{
    uint kase = 0;
    printf("Running tests on mat-vec multiplication...\n");
    while(kase < MAX_CASES)
    {
        memset(golden_v, 0, sizeof(golden_v));
        memset(A, 0, sizeof(A));
        memset(v, 0, sizeof(v));
        uint m = rand() % MAX_ROWS + 1;
        uint n = rand() % MAX_COLS + 1;
        uint nnz = rand() % MAX_NNZ + 1;
        rand_fill_mat(A, m, n, nnz);
        DynamicSparseMatrix spm;
        Vector calc_v(n);
        spm.resize(m, n);
        spm.setFromTriplets(tList, tList + nnz);
        for(uint i = 0; i < n; i++)
        {

        }
        if(!test_same(golden_v, calc_v))
        {
            std::cerr << "Failed test mat-vec-multiplication. Failing case is:" << std::endl;
            std::cerr << "A:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(A[i][j])) std::cerr << i << " " << j << " " << A[i][j] << std::endl;
            std::cerr << "B:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(B[i][j])) std::cerr << i << " " << j << " " << B[i][j] << std::endl;
            std::cerr << "std:" << std::endl;
            for(uint i = 0; i < m; i++)
                for(uint j = 0; j < n; j++)
                    if(!isZero(golden[i][j])) std::cerr << i << " " << j << " " << golden[i][j] << std::endl;
            std::cerr << "Your result is" << std::endl;
            std::cerr << "spm:" << std::endl << spm << std::endl;
            exit(-1);
        }
        printf("Passed test case %d\n", ++kase);
    }
    printf("Passed all tests on mat-vec multiplication.\n");
}

int main()
{
    srand(time(0));

    printf("Start testing arithmetics...\n");
    if constexpr (TESTS & TEST_SET)
        test_set();
    if constexpr (TESTS & TEST_ADD)
        test_add();
    if constexpr (TESTS & TEST_SUB)
        test_sub();
    printf("All tests passed!\n");
}
