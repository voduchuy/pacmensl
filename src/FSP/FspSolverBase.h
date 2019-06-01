//
// Created by Huy Vo on 5/29/18.
//

#ifndef PARALLEL_FSP_FSP_H
#define PARALLEL_FSP_FSP_H

#include<algorithm>
#include<cstdlib>
#include<cmath>
#include"Matrix/FspMatrixBase.h"
#include"OdeSolver/OdeSolverBase.h"
#include"FSS/StateSetBase.h"
#include"OdeSolver/cvode_interface/CVODEFSP.h"
#include"util/cme_util.h"

namespace cme {
    namespace parallel {
        struct FSPSolverComponentTiming {
            PetscReal StatePartitioningTime;
            PetscReal MatrixGenerationTime;
            PetscReal ODESolveTime;
            PetscReal SolutionScatterTime;
            PetscReal RHSEvalTime;
            PetscReal TotalTime;
        };

        class FspSolverBase {
            using Real = PetscReal;
            using Int = PetscInt;
        private:

            MPI_Comm comm = MPI_COMM_NULL;

            PartitioningType partitioning_type = Graph;
            PartitioningApproach repart_approach = FromScratch;
            ODESolverType odes_type = CVODE_BDF;

            bool custom_constraints = false;
            fsp_constr_multi_fn *fsp_constr_funs;
            arma::Row<int> fsp_bounds;
            arma::Row<Real> fsp_expasion_factors;

            StateSetBase *fsp;
            Vec *p;
            FspMatrixBase *A;
            OdeSolverBase *ode_solver;

            Real t_final = 0.0;
            Real fsp_tol = 0.0;

            arma::Mat<Int> stoich_mat;
            PropFun propensity;
            TcoefFun t_fun;

            std::function<void(PetscReal, Vec, Vec)> tmatvec;
            arma::Mat<Int> init_states;

            arma::Col<PetscReal> init_probs;
            int verbosity = 0;

            // For logging events using PETSc LogEvent
            PetscBool log_fsp_events = PETSC_FALSE;
            PetscLogEvent StateSetPartitioning;
            PetscLogEvent MatrixGeneration;
            PetscLogEvent ODESolve;
            PetscLogEvent SolutionScatter;
            PetscLogEvent RHSEvaluation;
            PetscLogEvent SettingUp;
            PetscLogEvent Solving;
        public:

            explicit FspSolverBase(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type);

            void SetFSPConstraintFunctions(fsp_constr_multi_fn *lhs_constr);

            void SetInitFSPBounds( arma::Row< int > &_fsp_size );

            void SetFinalTime(PetscReal t);

            void SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

            void SetFSPTolerance(PetscReal _fsp_tol);

            void SetStoichiometry(arma::Mat<Int> &stoich);

            void SetPropensity(PropFun _prop);

            void SetTimeFunc(TcoefFun _t_fun);

            void SetVerbosityLevel(int verbosity_level);

            void SetInitProbabilities(arma::Mat<Int> &_init_states, arma::Col<PetscReal> &_init_probs);

            void SetLogging(PetscBool logging);

            void SetFromOptions();

            void SetUp();

            Vec &GetP();

            StateSetBase *GetStateSubset();

            FSPSolverComponentTiming GetAvgComponentTiming();

            FiniteProblemSolverPerfInfo GetSolverPerfInfo();

            void Solve();

            void Destroy();

            ~FspSolverBase();

            friend StateSetBase;
            friend FspMatrixBase;
            friend OdeSolverBase;
        };
    }
}


#endif //PARALLEL_FSP_FSP_H
