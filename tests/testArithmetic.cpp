#include <SparseMatrix.h>
#include <random>
#include <ctime>
#include <spmx_utils.h>

using namespace SpmX;
Triplet tList[200];
const uint MAX_ROWS = 600, MAX_COLS = 800, MAX_NNZ = 300;
const int MAX_CASES = 100;
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

bool test_same(Real golden[][MAX_COLS], const DynamicSparseMatrix& spm)
{
    static std::vector<Triplet> v;
    v.clear();
    spm.toTriplets(v);
    v.reserve(spm.nonZeros());
    uint nnz = 0;
    for(uint i = 0; i < spm.rows(); i++)
        for(uint j = 0; j < spm.cols(); j++)
            if(!isZero(golden[i][j])) nnz++;
    if(nnz != spm.nonZeros())
    {
        std::cerr << "Testing setFromTriplets: wrong non-zeros" << std::endl;
        return false;
    }
    for(uint i = 0; i < spm.nonZeros(); i++)
        if(!isEqual(std::get<2>(v[i]), golden[std::get<0>(v[i])][std::get<1>(v[i])]))
        {
            std::cerr << "Testing setFromTriplets: wrong value" << std::endl;
            return false;
        }
    return true;
}

void test_set()
{
    static Real golden[MAX_ROWS][MAX_COLS];
    uint kase = 0;
    printf("Running tests on setFromTriplets...\n");
    while(kase < MAX_CASES)
    {
        memset(golden, 0, sizeof(golden));
        uint m = rand() % MAX_ROWS;
        uint n = rand() % MAX_COLS;
        uint nnz = rand() % MAX_NNZ;
        rand_fill_mat(golden, m, n, nnz);
        DynamicSparseMatrix spm;
        spm.resize(m, n);
        spm.setFromTriplets(tList, tList + nnz);
        if(!test_same(golden, spm))
        {
            std::cerr << "Failed test setFromTriplets" << std::endl;
            exit(-1);
        }
        kase++;
    }
    printf("Passed test setFromTriplets.\n");
}

int main()
{
    srand(time(0));
    printf("Start testing...\n");
    test_set();
    printf("Passed TestArithmetic.");
}
