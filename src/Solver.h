//
// Created by creeper on 23-3-5.
//

#ifndef SPMX_SOLVER_H
#define SPMX_SOLVER_H

#include "SparseMatrix.h"

namespace SpmX
{
    enum SolverStatus{ Undefined, Success, NumericalError, NotConverge };
    class FactorizeSolver
    {
        public:
            virtual void compute(const DynamicSparseMatrix& A) = 0;
            virtual Vector solve(const Vector& b) const = 0;
    };
    class IterativeSolver
    {
        public:
            SolverStatus info() const { return status; }
            virtual Vector solve(const Vector& b) const = 0;
        protected:
            SolverStatus status = Undefined;
    };
    class JacobiSolver final : public IterativeSolver
    {
        public:
            SolverStatus info() const { return status; }
            Vector solve(const Vector& b) const override;
    };
    class GaussSeidelSolver final : public IterativeSolver
    {
        public:
            SolverStatus info() const { return status; }
            Vector solve(const Vector& b) const override;
    };
    class CGSolver final : public IterativeSolver
    {
        public:
            SolverStatus info() const { return status; }
            Vector solve(const Vector& b) const override;
    };

}


#endif //SPMX_SOLVER_H
