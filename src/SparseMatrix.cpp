//
// Created by creeper on 23-3-5.
//
#include <SparseMatrix.h>
#include <cstring>
#include <spmx_utils.h>
#include <iostream>
#include "mySTL.h"

namespace SpmX
{
    /**
     * eliminate duplicates in the CSR storage of sparse matrix
     */
    void DynamicSparseMatrix::eliminateDuplicates()
    {
        if(storeType == CSR)
        {
            int *bucket = static_cast<int*>(malloc(sizeof(int) * n));
            memset(bucket, -1, sizeof(int) * n);
            for(uint i = 0; i < m; i++)
            {
                for(uint j = outer[i]; j < outer[i + 1]; j++)
                {
                    uint col_idx = inner[j];
                    if(bucket[col_idx] >= static_cast<int>(outer[i]))
                    {
                        val[bucket[col_idx]] += val[j];
                        inner[j] = ~0u; // mark the column of the duplicated places -1
                    }
                    else bucket[col_idx] = static_cast<int>(j);
                }
            }
            uint cnt = 0;
            for(uint i = 0; i < m; i++)
            {
                uint olderAi = outer[i];
                outer[i] -= cnt;
                for(uint j = olderAi; j < outer[i + 1]; j++)
                {
                    if(!(~inner[j])) cnt++;
                    else
                    {
                        inner[j - cnt] = inner[j];
                        val[j - cnt] = val[j];
                    }
                }
            }
            free(bucket);
            nnz -= cnt;
            outer[m] = nnz;
        }
    }
    /**
     * set the sparse matrix based on a consecutive sequence of triplets in CSR type
     * the m and n of spm must be set before, and the original space will be reallocated
     * this method isn't supposed to be called frequently
     * so to guarantee the property of the sparse matrix, refineStorage() will be called automatically
     * Complexity O(m + n + nnz)
     * @note repeated references are not allowed
     * @param spm pointer to the sparse matrix to be set
     * @param begin the beginning address of the sequence, closed
     * @param end the end address of the sequence, opened
     * @return whether the operation succeeds
     */
    void DynamicSparseMatrix::setFromTriplets(Triplet *begin, Triplet *end, StoreType type)
    {
        assert(end >= begin);
        storeType = type;
        if(inner) inner = static_cast<uint*>(realloc(inner, (end - begin) * sizeof(uint)));
        else inner = static_cast<uint*>(malloc((end - begin) * sizeof(uint)));
        if(val) val = static_cast<Real*>(realloc(val, (end - begin) * sizeof(Real)));
        else val = static_cast<Real*>(malloc((end - begin) * sizeof(Real)));
        nnz = 0;
        inOrder = true;
        if(type == CSR)
        {
            if(outer) outer = static_cast<uint*>(realloc(outer, (m + 1) * sizeof(uint)));
            else outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
            auto list = new List<std::tuple<uint, Real>>[m];
            for(Triplet *p = begin; p < end; p++)
            {
                if(isZero(std::get<2>(*p))) continue;
                list[std::get<0>(*p)].push_back({std::get<1>(*p), std::get<2>(*p)});
            }
            outer[0] = 0;
            for(uint i = 0; i < m; i++)
            {
                if(list[i].empty())
                {
                    outer[i + 1] = outer[i];
                    continue;
                }
                outer[i + 1] = outer[i] + list[i].size();
                for(std::tuple<uint, Real> tp : list[i])
                {
                    inner[nnz] = std::get<0>(tp);
                    val[nnz++] = std::get<1>(tp);
                }
            }
            for(uint i = 0; i <= m; i++)
                assert(outer[i] >= 0 && outer[i] <= nnz);
            delete[] list;
        }
        refineStorage();
        for(uint i = 0; i < m; i++)
        {
            for(uint j = outer[i] + 1; j < outer[i + 1]; j++)
            {
                if(inOrder && inner[j] < inner[j - 1])
                {
                    inOrder = false;
                    break;
                }
            }
        }
    }

    DynamicSparseMatrix DynamicSparseMatrix::operator+(const DynamicSparseMatrix &A) const
    {
        assert(m == A.m && n == A.n);
        assert(storeType == CSR && A.storeType == CSR);
        if(!A.inOrder) A.reOrder();
        if(!inOrder) reOrder();
        DynamicSparseMatrix ret(m, n);
        ret.outer = static_cast<uint*>(malloc(sizeof(uint) * (m + 1)));
        ret.inner = static_cast<uint*>(malloc(sizeof(uint) * (nnz + A.nnz)));
        ret.val = static_cast<Real*>(malloc(sizeof(Real) * (nnz + A.nnz)));
        uint j = 0, k = 0;
        for(uint i = 0; i < m; i++)
        {
            // the following algo merges the two ordered inner indices in linear time without additional space
            ret.outer[i] = ret.nnz;
            while(j < outer[i + 1] && k < A.outer[i + 1])
            {
                uint inner_idx = inner[j], inner_idx_A = A.inner[k];
                if(inner_idx < inner_idx_A)
                {
                    ret.inner[ret.nnz] = inner_idx;
                    ret.val[ret.nnz++] = val[j++];
                }
                else if(inner_idx == inner_idx_A)
                {
                    Real sum = 0.0;
                    while(j < outer[i + 1] && inner[j] == inner_idx)
                        sum += val[j++];
                    while(k < A.outer[i + 1] && A.inner[k] == inner_idx_A)
                        sum += A.val[k++];
                    if(!isZero(sum))
                    {
                        ret.inner[ret.nnz] = inner_idx;
                        ret.val[ret.nnz++] = sum;
                    }
                }
                else
                {
                    ret.inner[ret.nnz] = inner_idx_A;
                    ret.val[ret.nnz++] = A.val[k++];
                }
            }
            if(j == outer[i + 1])
            {
                while(k < A.outer[i + 1])
                {
                    ret.inner[ret.nnz] = A.inner[k];
                    ret.val[ret.nnz++] = A.val[k++];
                }
            }
            else
            {
                while(j < outer[i + 1])
                {
                    ret.inner[ret.nnz] = inner[j];
                    ret.val[ret.nnz++] = val[j++];
                }
            }
        }
        ret.outer[m] = ret.nnz;
        ret.inOrder = true;
        return ret;
    }

    /**
     * transpose the matrix, only supported for CSR style.
     * Complexity O(n + m + nnz)
     * @return the transpose of this matrix
     */
    DynamicSparseMatrix DynamicSparseMatrix::transpose() const
    {
        DynamicSparseMatrix ret(n, m);
        if(!nnz) return ret;
        ret.inner = static_cast<uint*>(malloc(nnz * sizeof(uint)));
        ret.val = static_cast<Real*>(malloc(nnz * sizeof(Real)));
        ret.nnz = nnz;
        ret.storeType = CSR;
        ret.outer = static_cast<uint*>(malloc((n + 1) * sizeof(uint)));
        uint *bucket = static_cast<uint*>(malloc(n * sizeof(uint)));
        memset(ret.outer, 0, sizeof(uint) * (n + 1));
        memset(bucket, 0, sizeof(uint) * n);
        for(uint i = 0; i < nnz; i++) ret.outer[inner[i] + 1]++;
        for(uint i = 1; i <= n; i++)
            ret.outer[i] += ret.outer[i - 1];
        for(uint i = 0; i < m; i++)
        {
            for(uint j = outer[i]; j < outer[i + 1]; j++)
            {
                uint idx = ret.outer[inner[j]] + bucket[inner[j]];
                ret.inner[idx] = i;
                ret.val[idx] = val[j];
                bucket[inner[j]]++;
            }
        }
        free(bucket);
        return ret;
    }

    DynamicSparseMatrix DynamicSparseMatrix::operator*(const DynamicSparseMatrix &A) const
    {
        assert(n == A.m);
        DynamicSparseMatrix ret(m, A.n);
        uint est_nnz = 0;
        // estimate the number of non-zero elements of the result
        for(uint i = 0; i < m; i++)
        {
            for(uint j = outer[i]; j < outer[i + 1]; j++)
            {
                uint idx = inner[j];
                for(uint k = A.outer[idx]; k < A.outer[idx + 1]; k++)
                    if(A.inner[k] == j) est_nnz++;
            }
        }
        if(!inOrder) reOrder();
        if(!A.inOrder) A.reOrder();
        ret.outer = static_cast<uint*>(malloc(sizeof(uint) * (ret.m + 1)));
        ret.inner = static_cast<uint*>(malloc(sizeof(uint) * est_nnz));
        ret.val = static_cast<Real*>(malloc(sizeof(Real) * est_nnz));
        for(uint i = 0; i < m; i++)
        {
            A.outer[i] = ret.nnz;
            for(uint j = outer[i]; j < outer[i + 1]; j++)
            {
                uint idx = inner[j];
                for(uint k = A.outer[idx]; k < A.outer[idx + 1]; k++)
                {
                    if(A.inner[k] == j)
                    {
                        ret.inner[ret.nnz] = j;
                        ret.val[ret.nnz] += val[j] * A.val[k];
                    }
                }
            }
        }
        return ret;
    }

    DynamicSparseMatrix DynamicSparseMatrix::operator-(const DynamicSparseMatrix &A) const
    {
        assert(m == A.m && n == A.n);
        assert(storeType == CSR && A.storeType == CSR);
        if(!A.inOrder) A.reOrder();
        if(!inOrder) reOrder();
        DynamicSparseMatrix ret(m, n);
        ret.outer = static_cast<uint*>(malloc(sizeof(uint) * (m + 1)));
        ret.inner = static_cast<uint*>(malloc(sizeof(uint) * (nnz + A.nnz)));
        ret.val = static_cast<Real*>(malloc(sizeof(Real) * (nnz + A.nnz)));
        uint j = 0, k = 0;
        for(uint i = 0; i < m; i++)
        {
            // the following algo merges the two ordered inner indices in linear time without additional space
            ret.outer[i] = ret.nnz;
            while(j < outer[i + 1] && k < A.outer[i + 1])
            {
                uint inner_idx = inner[j], inner_idx_A = A.inner[k];
                if(inner_idx < inner_idx_A)
                {
                    ret.inner[ret.nnz] = inner_idx;
                    ret.val[ret.nnz++] = val[j++];
                }
                else if(inner_idx == inner_idx_A)
                {
                    Real sum = 0.0;
                    while(j < outer[i + 1] && inner[j] == inner_idx)
                        sum += val[j++];
                    while(k < A.outer[i + 1] && A.inner[k] == inner_idx_A)
                        sum -= A.val[k++];
                    if(!isZero(sum))
                    {
                        ret.inner[ret.nnz] = inner_idx;
                        ret.val[ret.nnz++] = sum;
                    }
                }
                else
                {
                    ret.inner[ret.nnz] = inner_idx_A;
                    ret.val[ret.nnz++] = -A.val[k++];
                }
            }
            if(j == outer[i + 1])
            {
                while(k < A.outer[i + 1])
                {
                    ret.inner[ret.nnz] = A.inner[k];
                    ret.val[ret.nnz++] = -A.val[k++];
                }
            }
            else
            {
                while(j < outer[i + 1])
                {
                    ret.inner[ret.nnz] = inner[j];
                    ret.val[ret.nnz++] = val[j++];
                }
            }
        }
        ret.outer[m] = ret.nnz;
        ret.inOrder = true;
        return ret;
    }

    /**
     * Overload the operator << to output the sparse matrix.
     * By default the sparse matrix is output in a list of triplets.
     */
    std::ostream &operator<<(std::ostream &o, const DynamicSparseMatrix &spm)
    {
        try
        {
            if(spm.storeType == DynamicSparseMatrix::CSR)
            {
                if(!spm.nnz) return o;
                for(int i = 0; i < spm.m; i++)
                    for (uint j = spm.outer[i]; j < spm.outer[i + 1]; j++)
                        o << "(" << i << ", " << spm.inner[j] << ", " << spm.val[j] << ")" << std::endl;
            }
        }
        catch(std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }
        return o;
    }

    /**
     * transpose this matrix in-place
     * Complexity O(m + n + nnz)
     */
    void DynamicSparseMatrix::transposeInPlace()
    {
        if(!nnz)
        {
            std::swap(m, n);
            return;
        }
        uint *tmp_a = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
        uint *tmp_b = static_cast<uint*>(malloc(nnz * sizeof(uint)));
        Real *tmp_v = static_cast<Real*>(malloc(nnz * sizeof(Real)));
        memcpy(tmp_a, outer, sizeof(uint) * (m + 1));
        memcpy(tmp_b, inner, sizeof(uint) * nnz);
        memcpy(tmp_v, val, sizeof(Real) * nnz);
        outer = static_cast<uint*>(realloc(outer, (n + 1) * sizeof(uint)));
        uint *bucket = static_cast<uint*>(malloc(n * sizeof(uint)));
        memset(outer, 0, sizeof(uint) * (n + 1));
        memset(bucket, 0, sizeof(uint) * n);
        for(int i = 0; i < nnz; i++) outer[tmp_b[i] + 1]++;
        for(int i = 1; i <= n; i++)
            outer[i] += outer[i - 1];
        for(int i = 0; i < m; i++)
        {
            for(uint j = tmp_a[i]; j < tmp_a[i + 1]; j++)
            {
                uint idx = outer[tmp_b[j]] + bucket[tmp_b[j]];
                inner[idx] = i;
                val[idx] = tmp_v[j];
                bucket[tmp_b[j]]++;
            }
        }
        std::swap(m, n);
        free(tmp_a);
        free(tmp_b);
        free(tmp_v);
        free(bucket);
    }
    /**
     * put the storage of this matrix in order
     * Complexity O(nnz)
     */
    void DynamicSparseMatrix::reOrder() const
    {
        // transposing twice doesn't change the matrix
        // but transpose once does
        // so to call the non-const transpose I have to use some tricks
        auto *p = (DynamicSparseMatrix*)(this);
        p->transposeInPlace();
        p->transposeInPlace();
        inOrder = true;
    }

    /**
     * Eliminate the zeros(possibly caused by arithmetic operations) in the sparse matrix.
     * Since the occasions when arithmetic operations cause many zeros to occur are rare,
     * this method won't be called by default after arithmetic operations are done,
     * and it's leaved for the user to call manually.
     * However, you can call enableAutoEliminateZeros to call this method automatically after arithmetic operations
     * Complexity O(nnz)
     */
    void DynamicSparseMatrix::eliminateZeros()
    {
        uint cnt = 0;
        for(uint i = 0; i < m; i++)
        {
            uint olderAi = outer[i];
            outer[i] -= cnt;
            for(uint j = olderAi; j < outer[i + 1]; j++)
            {
                if(isZero(val[j])) cnt++;
                else if(cnt)
                {
                    inner[j - cnt] = inner[j];
                    val[j - cnt] = val[j];
                }
            }
        }
        nnz -= cnt;
        outer[m] = nnz;
    }

    void DynamicSparseMatrix::resize(uint _m, uint _n)
    {
        m = _m;
        n = _n;
    }

    Vector DynamicSparseMatrix::operator*(const Vector &V) const
    {
        assert(V.isCol() && n == V.dim());
        Vector ret;
        for(uint i = 0; i < m; i++)
        {
            ret[i] = 0.0;
            for(uint j = outer[i]; j < outer[i + 1]; j++)
                ret[i] += val[j] * V[inner[j]];
        }
        return ret;
    }

    Vector operator*(const Vector &V, const DynamicSparseMatrix &spm)
    {
        assert(V.isRow() && V.dim() == spm.m);
        Vector ret;
        for(uint i = 0; i < spm.m; i++)
        {
            ret[i] = 0.0;
            for(uint j = spm.outer[i]; j < spm.outer[i + 1]; j++)
                ret[i] += spm.val[j] * V[spm.inner[j]];
        }
        return ret;
    }
}