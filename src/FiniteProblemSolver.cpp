//
// Created by Huy Vo on 12/6/18.
//

#include <CVODEFSP.h>
#include <FiniteProblemSolver.h>

#include "FiniteProblemSolver.h"

void cme::parallel::FiniteProblemSolver::SetPrintIntermediateSteps(int iprint) {
    print_intermediate = iprint;
}

namespace cme{
    namespace parallel{

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
            rhs = std::move(_rhs);
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

        void FiniteProblemSolver::EnableLogging() {
            logging = PETSC_TRUE;
            perf_info.n_step = 0;
            perf_info.model_time.resize(100000);
            perf_info.cpu_time.resize(100000);
            perf_info.n_eqs.resize(100000);
        }

        FiniteProblemSolverPerfInfo FiniteProblemSolver::GetAvgPerfInfo() {
            assert(logging);

            FiniteProblemSolverPerfInfo perf_out = perf_info;

            PetscMPIInt comm_size;
            MPI_Comm_size(comm, &comm_size);

            for (auto i{perf_out.n_step-1}; i >= 0; --i){
                perf_out.cpu_time[i] = perf_out.cpu_time[i] - perf_out.cpu_time[0];
                MPI_Allreduce(MPI_IN_PLACE, (void*) &perf_out.cpu_time[i], 1, MPIU_REAL, MPI_SUM, comm);
            }

            for (auto i{0}; i < perf_out.n_step; ++i){
                perf_out.cpu_time[i] /= PetscReal(comm_size);
            }

            return perf_out;
        }
    }
}