//
// Created by Huy Vo on 12/6/18.
//

#include <CVODEFSP.h>
#include <FiniteProblemSolver.h>

#include "FiniteProblemSolver.h"

void cme::petsc::FiniteProblemSolver::SetPrintIntermediateSteps(int iprint) {
    print_intermediate = iprint;
}

namespace cme{
    namespace petsc{

        FiniteProblemSolver::FiniteProblemSolver(MPI_Comm new_comm) {
            MPI_Comm_dup(new_comm, &comm);
        }

        void FiniteProblemSolver::SetFinalTime(PetscReal _t_final) {
            t_final = _t_final;
        }

        void FiniteProblemSolver::SetFSPTolerance(PetscReal _fsp_tol) {
            fsp_tol = _fsp_tol;
        }

        void FiniteProblemSolver::SetInitSolution(Vec *_sol) {
            solution = _sol;
        }

        void FiniteProblemSolver::SetRHS(std::function<void (PetscReal, Vec, Vec)> _rhs) {
            rhs = _rhs;
        }

        void FiniteProblemSolver::SetFiniteStateSubset(FiniteStateSubset *_fsp) {
            fsp = _fsp;
            expand_sink.resize(fsp->GetNumSpecies());
        }

        void FiniteProblemSolver::RHSEval(PetscReal t, Vec x, Vec y) {
            rhs(t, x, y);
        }

        void FiniteProblemSolver::SetCurrentTime(PetscReal t) {
            t_now = t;
        }

        PetscReal FiniteProblemSolver::GetCurrentTime() {
            return t_now;
        }

        PetscInt FiniteProblemSolver::Solve() {
            // Make sure the necessary data has been set
            assert(solution != nullptr);
            assert(rhs);
            assert(fsp);
            return 0;
        }

        FiniteProblemSolver::~FiniteProblemSolver() {
            MPI_Comm_free(&comm);
            Free();
        }

        arma::Row<PetscInt> FiniteProblemSolver::GetExpansionIndicator() {
            return expand_sink;
        }
    }
}