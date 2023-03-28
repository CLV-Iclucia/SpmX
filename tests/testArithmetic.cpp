#include <SparseMatrix.h>
#include <random>
#include <ctime>
using namespace SpmX;
Triplet tList[200];
SpmX::DynamicSparseMatrix spm;

int main()
{
    srand(time(0));
    uint m = 1000;
    uint n = 1200;
    uint nnz = 100;
    for(int i = 0; i < nnz; i++)
    {
        uint x = rand() % m;
        uint y = rand() % n;
        Real val = 1.0 * rand() / (rand() + 1.0);
        tList[i] = {x, y, val};
        std::cout << x << " " << y << " " << val << std::endl;
     }
    spm.resize(m, n);
    spm.setFromTriplets(tList, tList + nnz);
}
