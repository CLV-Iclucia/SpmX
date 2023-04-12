//
// Created by creeper on 23-3-5.
//

#ifndef SPMX_SOLVER_H
#define SPMX_SOLVER_H

#include "SparseMatrix.h"

namespace SpmX
{
    enum SolverStatus{ Undefined, Success, NumericalError, NotConverge };
    template<typename Derived>
    class FactorizeSolver
    {
        protected:
            SolverStatus status = Undefined;
        public:
            SolverStatus info() const { return status; }
            void compute(const SparseMatrix& A) { static_cast<Derived*>(this)->compute(A); };
            Vector solve(const Vector& b) const { static_cast<Derived*>(this)->solve(b); };
    };

    template<typename Derived>
    class IterativeSolver
    {
        public:
            virtual Vector solve(const Vector& b) const = 0;
        protected:
            SolverStatus status = Undefined;
    };
    class JacobiSolver final : public IterativeSolver<JacobiSolver>
    {
        public:
            Vector solve(const Vector& b) const;
    };
    class GaussSeidelSolver final : public IterativeSolver<GaussSeidelSolver>
    {
        public:
            Vector solve(const Vector& b) const;
    };
    class CGSolver final : public IterativeSolver<CGSolver>
    {
        public:
            Vector solve(const Vector& b) const;
    };

}


#endif //SPMX_SOLVER_H
