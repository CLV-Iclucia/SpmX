//
// Created by creeper on 23-3-24.
//

#ifndef SPMX_SPMX_VECTOR_H
#define SPMX_SPMX_VECTOR_H

#include <cstdlib>
#include <spmx_types.h>
#include <cstring>
#include <cassert>

namespace SpmX
{
    class Vector
    {
        private:
            enum {Col, Row} type = Col;
            Real *a = nullptr;
            uint n = 0;
        public:
            bool isRow() const { return type == Row; }
            bool isCol() const { return type == Col; }
            Vector() = default;
            Vector(Real vec[], uint _dim) : n(_dim) { memcpy(a, vec, sizeof(Real) * _dim); }
            Vector(const Vector& v) : type(v.type), n(v.n)
            {
                a = static_cast<Real*>(malloc(n * sizeof(Real)));
                memcpy(a, v.a, n * sizeof(Real));
            }
            Vector(Vector&& v) noexcept : n(v.n), a(v.a) { v.a = nullptr; }
            explicit Vector(uint _n) : n(_n)
            {
                a = static_cast<Real*>(malloc(n * sizeof(Real)));
                memset(a, 0, sizeof(Real) * n);
            }
            Real operator[](uint idx) const
            {
                assert(idx < n);
                return *(a + idx);
            }
            Real& operator[](uint idx)
            {
                assert(idx < n);
                return *(a + idx);
            }
            uint dim() const { return n; }
            Vector transpose() const
            {
                Vector ret(*this);
                if(ret.type == Col) ret.type = Row;
                else ret.type = Col;
                return ret;
            }
            void transposeInPlace()
            {
                if(type == Col) type = Row;
                else type = Col;
            }
            ~Vector() { free(a); }
    };

}


#endif //SPMX_SPMX_VECTOR_H
