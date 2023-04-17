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
#include "spmx_utils.h"
#include "mySTL.h"

namespace SpmX
{
    template<bool MemStrategy> class SparseMatrix;
    template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
    SparseMatrix<RetDynamic> SparseMatrixAdd(const SparseMatrix<LhsDynamic>& A, const SparseMatrix<RhsDynamic>& B);
    template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
    SparseMatrix<RetDynamic> SparseMatrixMinus(const SparseMatrix<LhsDynamic>& A, const SparseMatrix<RhsDynamic>& B);
    /**
      * Beside dynamic sparse matrix, specifically designed static sparse matrix is also provided.
      * When constructing static sparse matrix, the shape of the matrix must be indicated,
      * and the elements must be given in a Triplet array.
      * When it is created, the storage will be refined automatically for future use.
      * Such matrix is particularly useful for physically based animation, where the matrix will not change after created.
      * And the operations can be optimized specifically.
      */
    template<>
    class SparseMatrix<Static>
    {
            template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
            friend SparseMatrix<RetDynamic> SparseMatrixAdd(const SparseMatrix<LhsDynamic>& A, const SparseMatrix<RhsDynamic>& B);
            template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
            friend SparseMatrix<RetDynamic> SparseMatrixMinus(const SparseMatrix<LhsDynamic>& A, const SparseMatrix<RhsDynamic>& B);
        private:
            uint m = 0, n = 0, nnz = 0;
            uint *outer = nullptr, *inner = nullptr;
            Real *val = nullptr;
            SparseMatrix(uint _m, uint _n) { m = _m; n = _n; }
            void reOrder() const
            {
                auto *p = (SparseMatrix*)(this);
                p->transposeInPlace();
                p->transposeInPlace();
            }
            void eliminateDuplicates()
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
            /**
             * Eliminate the zeros(possibly caused by arithmetic operations) in the sparse matrix.
             * Since the occasions when arithmetic operations cause many zeros to occur are rare,
             * this method won't be called by default after arithmetic operations are done,
             * and it's leaved for the user to call manually.
             * However, you can call enableAutoEliminateZeros to call this method automatically after arithmetic operations
             * Complexity O(nnz)
             */
            void eliminateZeros()
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
            void transposeInPlace()
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
        public:
            template<bool Any>
            explicit SparseMatrix(SparseMatrix<Any>&& A) noexcept : m(A.m), n(A.n), nnz(A.nnz), outer(A.outer),
                                                                    inner(A.inner), val(A.val)
            {
                A.outer = A.inner = A.val = nullptr;
            }
            template<bool Any>
            explicit SparseMatrix(const SparseMatrix<Any>& A) : m(A.m), n(A.n), nnz(A.nnz)
            {
                outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
                inner = static_cast<uint*>(malloc(nnz * sizeof(uint)));
                val = static_cast<Real*>(malloc(nnz * sizeof(Real)));
                memcpy(outer, A.outer, (m + 1) * sizeof(uint));
                memcpy(inner, A.inner, nnz * sizeof(uint));
                memcpy(val, A.iner, nnz * sizeof(Real));
            }
            template<bool Ordered = false, bool NoDuplicates = false, bool NoZeros = false>
            SparseMatrix(uint _m, uint _n, Triplet* begin, Triplet* end) : m(_m), n(_n)
            {
                assert(end >= begin);
                inner = static_cast<uint*>(malloc((end - begin) * sizeof(uint)));
                val = static_cast<Real*>(malloc((end - begin) * sizeof(Real)));
                nnz = 0;
                outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
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
                delete[] list;
                if constexpr (!NoDuplicates) eliminateDuplicates();
                if constexpr (!NoZeros) eliminateZeros();
                if constexpr (!Ordered)
                {
                    bool inOrder = true;
                    for (uint i = 0; i < m; i++)
                    {
                        for (uint j = outer[i] + 1; j < outer[i + 1]; j++)
                        {
                            if (inOrder && inner[j] < inner[j - 1])
                            {
                                inOrder = false;
                                break;
                            }
                        }
                    }
                    reOrder();
                }
            }
            /**
             * Overload the operator << to output the sparse matrix.
             * By default the sparse matrix is output in a list of triplets.
             */
            friend std::ostream& operator<<(std::ostream& o, const SparseMatrix& spm)
            {
                try
                {
                    if(!spm.nnz) return o;
                    for(int i = 0; i < spm.m; i++)
                        for (uint j = spm.outer[i]; j < spm.outer[i + 1]; j++)
                            o << "(" << i << ", " << spm.inner[j] << ", " << spm.val[j] << ")" << std::endl;
                }
                catch(std::exception &e)
                {
                    std::cerr << e.what() << std::endl;
                }
                return o;
            }
            friend Vector operator*(const Vector& V, const SparseMatrix& spm)
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
            SparseMatrix() = default;
            uint rows() const { return m; }
            uint cols() const { return n; }
            uint nonZeros() const { return nnz; }
            template<bool Any>
            SparseMatrix operator+(const SparseMatrix<Any>& A) const
            {
                assert(m == A.m && n == A.n);
                return SparseMatrixAdd<Static, Static, Any>((*this), A);
            }
            SparseMatrix operator*(const SparseMatrix& A) const
            {
                assert(n == A.m);
                SparseMatrix ret(m, A.n);
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
            template<bool Any>
            SparseMatrix operator-(const SparseMatrix<Any>& A) const
            {
                assert(m == A.m && n == A.n);
                return SparseMatrixMinus<Static, Static, Any>((*this), A);
            }
            Vector operator*(const Vector& v) const
            {
                assert(v.isCol() && n == v.dim());
                Vector ret;
                for(uint i = 0; i < m; i++)
                {
                    ret[i] = 0.0;
                    for(uint j = outer[i]; j < outer[i + 1]; j++)
                        ret[i] += val[j] * v[inner[j]];
                }
                return ret;
            }
            SparseMatrix transpose() const
            {
                SparseMatrix ret(n, m);
                if (!nnz) return ret;
                ret.inner = static_cast<uint *>(malloc(nnz * sizeof(uint)));
                ret.val = static_cast<Real *>(malloc(nnz * sizeof(Real)));
                ret.nnz = nnz;
                ret.outer = static_cast<uint *>(malloc((n + 1) * sizeof(uint)));
                uint *bucket = static_cast<uint *>(malloc(n * sizeof(uint)));
                memset(ret.outer, 0, sizeof(uint) * (n + 1));
                memset(bucket, 0, sizeof(uint) * n);
                for (uint i = 0; i < nnz; i++) ret.outer[inner[i] + 1]++;
                for (uint i = 1; i <= n; i++)
                    ret.outer[i] += ret.outer[i - 1];
                for (uint i = 0; i < m; i++)
                {
                    for (uint j = outer[i]; j < outer[i + 1]; j++)
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

            void toTriplets(std::vector<Triplet>& v) const
            {
                for(uint i = 0; i < m; i++)
                    for(uint j = outer[i]; j < outer[i + 1]; j++)
                        v.emplace_back(i, inner[j], val[j]);
            }
            ~SparseMatrix() { free(outer); free(inner); free(val); }
    };

    template<bool MemStrategy = Dynamic>
    class SparseMatrix
    {
            template<bool> friend class SparseMatrix;
            template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
            friend SparseMatrix<RetDynamic> SparseMatrixAdd(const SparseMatrix<LhsDynamic>& A, const SparseMatrix<RhsDynamic>& B);
            template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
            friend SparseMatrix<RetDynamic> SparseMatrixMinus(const SparseMatrix<LhsDynamic>& A, const SparseMatrix<RhsDynamic>& B);
        private:
            mutable enum StoreType{CSR, COO} storeType = CSR;///< the way the sparse matrix is stored. By default it is CSR.
            uint m = 0, n = 0, nnz = 0;
            mutable uint *outer = nullptr, *inner = nullptr;
            mutable Real *val = nullptr;
            mutable bool inOrder = false;
            ///< whether the matrix is ordered, by default it's false.
            ///< When calling setFromTriplets, the order of the data will be checked and this field will be set.
            ///< Before operations this field will be checked. And reOrder will be called if it's false.
            void reOrder() const
            {
                auto *p = (SparseMatrix*)(this);
                p->transposeInPlace();
                p->transposeInPlace();
                inOrder = true;
            }
        public:
            SparseMatrix(uint _m, uint _n, uint _nnz) : m(_m), n(_n), nnz(_nnz) { }
            template<bool Any>
            explicit SparseMatrix(SparseMatrix<Any>&& A) noexcept : m(A.m), n(A.n), nnz(A.nnz), outer(A.outer),
                                                             inner(A.inner), val(A.val)
            {
                A.outer = A.inner = A.val = nullptr;
                if constexpr (Any == Static)
                    inOrder = true;
                else inOrder = A.inOrder;
            }
            template<bool Any>
            explicit SparseMatrix(const SparseMatrix<Any>& A) : m(A.m), n(A.n), inOrder(A.inOrder), nnz(A.nnz)
            {
                outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
                inner = static_cast<uint*>(malloc(nnz * sizeof(uint)));
                val = static_cast<Real*>(malloc(nnz * sizeof(Real)));
                memcpy(outer, A.outer, (m + 1) * sizeof(uint));
                memcpy(inner, A.inner, nnz * sizeof(uint));
                memcpy(val, A.iner, nnz * sizeof(Real));
            }

            /**
             * Overload the operator << to output the sparse matrix.
             * By default the sparse matrix is output in a list of triplets.
             */
            friend std::ostream& operator<<(std::ostream& o, const SparseMatrix& spm)
            {
                try
                {
                    if(spm.storeType == SparseMatrix::CSR)
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
            friend Vector operator*(const Vector& V, const SparseMatrix& spm)
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
            SparseMatrix() = default;
            SparseMatrix(uint _m, uint _n) : m(_m), n(_n) {}
            uint rows() const { return m; }
            uint cols() const { return n; }
            uint nonZeros() const { return nnz; }
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
             */
            void setFromTriplets(Triplet *begin, Triplet *end, StoreType type = CSR)
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
                if(!inOrder) reOrder();
            }
            template<bool Any>
            SparseMatrix& operator=(const SparseMatrix<Any>& A)
            {
                if constexpr(Any == Dynamic)
                {
                    if(&A == this) return *this;
                    inOrder = A.inOrder;
                }
                m = A.m;
                n = A.n;
                nnz = A.nnz;
                if(outer) outer = static_cast<uint*>(realloc(outer, (m + 1) * sizeof(uint)));
                else outer = static_cast<uint*>(malloc((m + 1) * sizeof(uint)));
                if(inner) inner = static_cast<uint*>(realloc(inner, nnz * sizeof(uint)));
                else inner = static_cast<uint*>(malloc(nnz * sizeof(uint)));
                if(val) val = static_cast<Real*>(realloc(val, nnz * sizeof(Real)));
                else val = static_cast<Real*>(malloc(nnz * sizeof(Real)));
                memcpy(outer, A.outer, (m + 1) * sizeof(uint));
                memcpy(inner, A.inner, nnz * sizeof(uint));
                memcpy(val, A.iner, nnz * sizeof(Real));
                return *this;
            }
            template<bool Any>
            SparseMatrix& operator=(SparseMatrix<Any>&& A) noexcept
            {
                m = A.m;
                n = A.n;
                nnz = A.nnz;
                if constexpr(Any == Dynamic) inOrder = A.inOrder;
                else inOrder = true;
                outer = A.outer;
                A.outer = nullptr;
                inner = A.inner;
                A.inner = nullptr;
                val = A.val;
                A.val = nullptr;
                return *this;
            }
            template<bool Any>
            SparseMatrix operator+(const SparseMatrix<Any>& A) const
            {
                assert(m == A.m && n == A.n);
                assert(storeType == CSR && A.storeType == CSR);
                if(!A.inOrder) A.reOrder();
                if(!inOrder) reOrder();
                return SparseMatrixAdd<Static, Dynamic, Any>((*this), A);
            }
            SparseMatrix operator*(const SparseMatrix& A) const
            {
                assert(n == A.m);
                SparseMatrix ret(m, A.n);
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
            template<bool Any>
            SparseMatrix operator-(const SparseMatrix<Any>& A) const
            {
                assert(m == A.m && n == A.n);
                assert(storeType == CSR && A.storeType == CSR);
                if(!inOrder) reOrder();
                if constexpr (Any == Dynamic)
                {
                    if(!inOrder) A.reOrder();
                }
                SparseMatrix ret(m, n);
                return SparseMatrixMinus<Static, Dynamic, Any>((*this), A);
            }
            Vector operator*(const Vector& v) const
            {
                assert(v.isCol() && n == v.dim());
                Vector ret;
                for(uint i = 0; i < m; i++)
                {
                    ret[i] = 0.0;
                    for(uint j = outer[i]; j < outer[i + 1]; j++)
                        ret[i] += val[j] * v[inner[j]];
                }
                return ret;
            }
            SparseMatrix transpose() const
            {
                SparseMatrix ret(n, m);
                if (!nnz) return ret;
                ret.inner = static_cast<uint *>(malloc(nnz * sizeof(uint)));
                ret.val = static_cast<Real *>(malloc(nnz * sizeof(Real)));
                ret.nnz = nnz;
                ret.storeType = CSR;
                ret.outer = static_cast<uint *>(malloc((n + 1) * sizeof(uint)));
                uint *bucket = static_cast<uint *>(malloc(n * sizeof(uint)));
                memset(ret.outer, 0, sizeof(uint) * (n + 1));
                memset(bucket, 0, sizeof(uint) * n);
                for (uint i = 0; i < nnz; i++) ret.outer[inner[i] + 1]++;
                for (uint i = 1; i <= n; i++)
                    ret.outer[i] += ret.outer[i - 1];
                for (uint i = 0; i < m; i++)
                {
                    for (uint j = outer[i]; j < outer[i + 1]; j++)
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
            void resize(uint _m, uint _n) { m = _m; n = _n; }
            void transposeInPlace()
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
            void eliminateDuplicates()
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
             * Eliminate the zeros(possibly caused by arithmetic operations) in the sparse matrix.
             * Since the occasions when arithmetic operations cause many zeros to occur are rare,
             * this method won't be called by default after arithmetic operations are done,
             * and it's leaved for the user to call manually.
             * However, you can call enableAutoEliminateZeros to call this method automatically after arithmetic operations
             * Complexity O(nnz)
             */
            void eliminateZeros()
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
            ~SparseMatrix() { free(outer); free(inner); free(val); }
    };

    template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
    SparseMatrix<RetDynamic> SparseMatrixAdd(const SparseMatrix<LhsDynamic> &A, const SparseMatrix<RhsDynamic> &B)
    {
        SparseMatrix<RetDynamic> ret(A.m, A.n);
        ret.outer = static_cast<uint*>(malloc(sizeof(uint) * (A.m + 1)));
        uint j = 0, k = 0;
        // the first pass calculate the non-zeros of the new matrix
        for(uint i = 0; i < A.m; i++)
        {
            ret.outer[i] = ret.nnz;
            while(j < A.outer[i + 1] && k < B.outer[i + 1])
            {
                uint inner_idx_A = A.inner[j], inner_idx_B = B.inner[k];
                if(inner_idx_A < inner_idx_B)
                {
                    ret.nnz++;
                    j++;
                }
                else if(inner_idx_A == inner_idx_B)
                {
                    Real sum = 0.0;
                    while(j < A.outer[i + 1] && A.inner[j] == inner_idx_A)
                        sum += A.val[j++];
                    while(k < B.outer[i + 1] && B.inner[k] == inner_idx_B)
                        sum += B.val[k++];
                    if(!isZero(sum)) ret.nnz++;
                }
                else
                {
                    ret.nnz++;
                    k++;
                }
            }
            if(j == A.outer[i + 1])
            {
                ret.nnz += B.outer[i + 1] - k;
                k = B.outer[i + 1];
            }
            else
            {
                ret.nnz += A.outer[i + 1] - j;
                j = A.outer[i + 1];
            }
        }
        ret.outer[A.m] = ret.nnz;
        ret.inner = static_cast<uint*>(malloc(sizeof(uint) * ret.nnz));
        ret.val = static_cast<Real*>(malloc(sizeof(Real) * ret.nnz));
        j = k = 0;
        uint ptr = 0;
        for(uint i = 0; i < A.m; i++)
        {
            while(j < A.outer[i + 1] && k < B.outer[i + 1])
            {
                uint inner_idx_A = A.inner[j], inner_idx_B = B.inner[k];
                if(inner_idx_A < inner_idx_B)
                {
                    ret.inner[ptr] = inner_idx_A;
                    ret.val[ptr++] = A.val[j++];
                }
                else if(inner_idx_A == inner_idx_B)
                {
                    Real sum = 0.0;
                    while(j < A.outer[i + 1] && A.inner[j] == inner_idx_A)
                        sum += A.val[j++];
                    while(k < B.outer[i + 1] && B.inner[k] == inner_idx_B)
                        sum += B.val[k++];
                    if(!isZero(sum))
                    {
                        ret.inner[ptr] = inner_idx_A;
                        ret.val[ptr++] = sum;
                    }
                }
                else
                {
                    ret.inner[ptr] = inner_idx_B;
                    ret.val[ptr++] = B.val[k++];
                }
            }
            if(j == A.outer[i + 1])
            {
                while(k < B.outer[i + 1])
                {
                    ret.inner[ptr] = B.inner[k];
                    ret.val[ptr++] = B.val[k++];
                }
            }
            else
            {
                while(j < A.outer[i + 1])
                {
                    ret.inner[ptr] = A.inner[j];
                    ret.val[ptr++] = A.val[j++];
                }
            }
        }
        return ret;
    }

    template<bool RetDynamic, bool LhsDynamic, bool RhsDynamic>
    SparseMatrix<RetDynamic> SparseMatrixMinus(const SparseMatrix<LhsDynamic> &A, const SparseMatrix<RhsDynamic> &B)
    {
        SparseMatrix<RetDynamic> ret(A.m, A.n);
        ret.outer = static_cast<uint*>(malloc(sizeof(uint) * (A.m + 1)));
        uint j = 0, k = 0;
        // the first pass calculate the non-zeros of the new matrix
        for(uint i = 0; i < A.m; i++)
        {
            ret.outer[i] = ret.nnz;
            while(j < A.outer[i + 1] && k < B.outer[i + 1])
            {
                uint inner_idx_A = A.inner[j], inner_idx_B = B.inner[k];
                if(inner_idx_A < inner_idx_B)
                {
                    ret.nnz++;
                    j++;
                }
                else if(inner_idx_A == inner_idx_B)
                {
                    Real sum = 0.0;
                    while(j < A.outer[i + 1] && A.inner[j] == inner_idx_A)
                        sum += A.val[j++];
                    while(k < B.outer[i + 1] && B.inner[k] == inner_idx_B)
                        sum -= B.val[k++];
                    if(!isZero(sum)) ret.nnz++;
                }
                else
                {
                    ret.nnz++;
                    k++;
                }
            }
            if(j == A.outer[i + 1])
            {
                ret.nnz += B.outer[i + 1] - k;
                k = B.outer[i + 1];
            }
            else
            {
                ret.nnz += A.outer[i + 1] - j;
                j = A.outer[i + 1];
            }
        }
        ret.outer[A.m] = ret.nnz;
        ret.inner = static_cast<uint*>(malloc(sizeof(uint) * ret.nnz));
        ret.val = static_cast<Real*>(malloc(sizeof(Real) * ret.nnz));
        j = k = 0;
        uint ptr = 0;
        for(uint i = 0; i < A.m; i++)
        {
            while(j < A.outer[i + 1] && k < B.outer[i + 1])
            {
                uint inner_idx_A = A.inner[j], inner_idx_B = B.inner[k];
                if(inner_idx_A < inner_idx_B)
                {
                    ret.inner[ptr] = inner_idx_A;
                    ret.val[ptr++] = A.val[j++];
                }
                else if(inner_idx_A == inner_idx_B)
                {
                    Real sum = 0.0;
                    while(j < A.outer[i + 1] && A.inner[j] == inner_idx_A)
                        sum += A.val[j++];
                    while(k < B.outer[i + 1] && B.inner[k] == inner_idx_B)
                        sum -= B.val[k++];
                    if(!isZero(sum))
                    {
                        ret.inner[ptr] = inner_idx_A;
                        ret.val[ptr++] = sum;
                    }
                }
                else
                {
                    ret.inner[ptr] = inner_idx_B;
                    ret.val[ptr++] = -B.val[k++];
                }
            }
            if(j == A.outer[i + 1])
            {
                while(k < B.outer[i + 1])
                {
                    ret.inner[ptr] = -B.inner[k];
                    ret.val[ptr++] = -B.val[k++];
                }
            }
            else
            {
                while(j < A.outer[i + 1])
                {
                    ret.inner[ptr] = A.inner[j];
                    ret.val[ptr++] = A.val[j++];
                }
            }
        }
        return ret;
    }
}


#endif //SPMX_SPARSEMATRIX_H
