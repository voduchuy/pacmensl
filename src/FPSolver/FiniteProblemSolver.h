//
// Created by Huy Vo on 12/6/18.
//

#ifndef PARALLEL_FSP_FINITEPROBLEMSOLVER_H
#define PARALLEL_FSP_FINITEPROBLEMSOLVER_H

#include "util/cme_util.h"
#include "FSS/FiniteStateSubset.h"

namespace cme{
    namespace parallel{
        enum ODESolverType {Magnus4, CVODE_BDF};

        struct FiniteProblemSolverPerfInfo{
            PetscInt n_step;
            std::vector<PetscInt> n_eqs;
            std::vector<PetscLogDouble> cpu_time;
            std::vector<PetscReal> model_time;
        };

        class FiniteProblemSolver {
        protected:
            MPI_Comm comm = MPI_COMM_NULL;
            FiniteStateSubset *fsp = nullptr;
            arma::Row<PetscInt> expand_sink;
            Vec *solution = nullptr;
            std::function<void (PetscReal t, Vec x, Vec y)> rhs;
            PetscReal t_now = 0.0;
            PetscReal t_final = 0.0;
            PetscReal fsp_tol = 0.0;
            ODESolverType solver_type;

            // For logging and monitoring
            int print_intermediate = 0;

            PetscBool logging = PETSC_FALSE;
            FiniteProblemSolverPerfInfo perf_info;
        public:
            explicit FiniteProblemSolver(MPI_Comm new_comm);

            void SetFiniteStateSubset(FiniteStateSubset *_fsp);
            void SetFinalTime(PetscReal _t_final);
            void SetFSPTolerance(PetscReal _fsp_tol);
            void SetInitSolution(Vec *sol0);
            void SetRHS(std::function<void (PetscReal, Vec, Vec)>_rhs);
            void SetCurrentTime(PetscReal t);
            void SetPrintIntermediateSteps(int iprint);
            void EnableLogging();

            void RHSEval(PetscReal t, Vec x, Vec y);

            virtual PetscInt Solve(); // Advance the solution toward final time. Return 0 if reaching final time, 1 if the FSP criteria fails before reaching final time.
            PetscReal GetCurrentTime();
            FiniteProblemSolverPerfInfo GetAvgPerfInfo();

            arma::Row<PetscInt> GetExpansionIndicator();
            virtual void Free(){};

            ~FiniteProblemSolver();
        };
    }
}

#endif //PARALLEL_FSP_FINITEPROBLEMSOLVER_H
