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
            DynamicSparseMatrix(uint _m, uint _n, uint _nnz) : m(_m), n(_n), nnz(_nnz) { }
            DynamicSparseMatrix(DynamicSparseMatrix&& A) noexcept : m(A.m), n(A.n), nnz(A.nnz), outer(A.outer),
                inner(A.inner), val(A.val), inOrder(A.inOrder)
                {
                    A.outer = A.inner = nullptr;
                    A.val = nullptr;
                }
            DynamicSparseMatrix(const DynamicSparseMatrix& A) : m(A.m), n(A.n), inOrder(A.inOrder), nnz(A.nnz)
            {
                outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
                inner = static_cast<uint*>(malloc(nnz * sizeof(uint)));
                val = static_cast<Real*>(malloc(nnz * sizeof(Real)));
            }
            friend std::ostream& operator<<(std::ostream& o, const DynamicSparseMatrix& spm);
            friend Vector operator*(const Vector& V, const DynamicSparseMatrix& spm);
            DynamicSparseMatrix() = default;
            DynamicSparseMatrix(uint _m, uint _n) : m(_m), n(_n) {}
            uint rows() const { return m; }
            uint cols() const { return n; }
            uint nonZeros() const { return nnz; }
            void setFromTriplets(Triplet *begin, Triplet *end, StoreType type = CSR);
            DynamicSparseMatrix& operator=(const DynamicSparseMatrix& A)
            {
                if(&A == this) return *this;
                else
                {
                    m = A.m;
                    n = A.n;
                    nnz = A.nnz;
                    if(outer) outer = static_cast<uint*>(realloc(outer, (m + 1) * sizeof(uint)));
                    else outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
                    if(inner) inner = static_cast<uint*>(realloc(inner, nnz * sizeof(uint)));
                    else inner = static_cast<uint*>(malloc(nnz * sizeof(uint)));
                    if(val) val = static_cast<Real*>(realloc(val, nnz * sizeof(Real)));
                    else val = static_cast<Real*>(malloc(nnz * sizeof(Real)));
                    inOrder = A.inOrder;
                    return *this;
                }
            }
            DynamicSparseMatrix& operator=(DynamicSparseMatrix&& A) noexcept
            {
                m = A.m;
                n = A.n;
                nnz = A.nnz;
                inOrder = A.inOrder;
                outer = A.outer;
                A.outer = nullptr;
                inner = A.inner;
                A.inner = nullptr;
                val = A.val;
                A.val = nullptr;
                return *this;
            }
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
                //if(!inOrder) reOrder();
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
