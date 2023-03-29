//
// Created by creeper on 23-2-23.
//

#ifndef SPMX_SPARSEMATRIX_H
#define SPMX_SPARSEMATRIX_H

#include <cstring>
#include <cassert>
#include <iostream>
#include <spmx_types.h>
#include <spmx_options.h>
#include <spmx_Vector.h>
#include <vector>

namespace SpmX
{
    class DynamicSparseMatrix;
    class DynamicSparseMatrix
    {
        private:
            mutable enum StoreType{CSR} storeType = CSR;///< the way the sparse matrix is stored. By default it is CSR.
            uint m = 0, n = 0, nnz = 0;
            mutable uint *outer = nullptr, *inner = nullptr;
            mutable Real *val = nullptr;
            mutable bool inOrder = false;
            ///< whether the matrix is ordered, by default it's false.
            ///< When calling setFromTriplets, the order of the data will be checked and this field will be set.
            ///< Before operations this field will be checked. And reOrder will be called if it's false.
            void reOrder() const;
        public:
            friend std::ostream& operator<<(std::ostream& o, const DynamicSparseMatrix& spm);
            friend Vector operator*(const Vector& V, const DynamicSparseMatrix& spm);
            DynamicSparseMatrix() = default;
            DynamicSparseMatrix(uint _m, uint _n) : m(_m), n(_n) {}
            uint rows() const { return m; }
            uint cols() const { return n; }
            uint nonZeros() const { return nnz; }
            void setFromTriplets(Triplet *begin, Triplet *end, StoreType type = CSR);
            DynamicSparseMatrix operator+(const DynamicSparseMatrix& A) const;
            DynamicSparseMatrix operator*(const DynamicSparseMatrix& A) const;
            DynamicSparseMatrix operator-(const DynamicSparseMatrix& A) const;
            Vector operator*(const Vector& v) const;
            DynamicSparseMatrix transpose() const;
            void resize(uint _m, uint _n);
            void transposeInPlace();
            void eliminateDuplicates();
            void eliminateZeros();
            void refineStorage()
            {
                eliminateDuplicates();
                eliminateZeros();
            }
            void toTriplets(std::vector<Triplet>& v) const
            {
                if(!inOrder) reOrder();
                for(uint i = 0; i < m; i++)
                    for(uint j = outer[i]; j < outer[i + 1]; j++)
                        v.emplace_back(i, inner[j], val[j]);
            }
            ~DynamicSparseMatrix() { free(outer); free(inner); free(val); }
    };

    class SymmetricSparseMatrix
    {

    };


}


#endif //SPMX_SPARSEMATRIX_H
