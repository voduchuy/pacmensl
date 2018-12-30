//
// Created by Huy Vo on 5/29/18.
//

#ifndef PARALLEL_FSP_FSP_H
#define PARALLEL_FSP_FSP_H

#include<algorithm>
#include<cstdlib>
#include<cmath>
#include"Matrix/MatrixSet.h"
#include"FPSolver/FiniteProblemSolver.h"
#include"FSS/FiniteStateSubset.h"
#include"FSS/FiniteStateSubsetNaive.h"
#include"FSS/FiniteStateSubsetRCB.h"
#include"FSS/FiniteStateSubsetGraph.h"
#include"FSS/FiniteStateSubsetHyperGraph.h"
#include"FSS/FiniteStateSubsetHierarch.h"
#include"FPSolver/CVODEFSP.h"
#include"util/cme_util.h"

namespace cme {
    namespace parallel {
        struct FSPSolverComponentTiming {
            PetscReal StatePartitioningTime;
            PetscReal MatrixGenerationTime;
            PetscReal ODESolveTime;
            PetscReal SolutionScatterTime;
            PetscReal RHSEvalTime;
        };

        class FSPSolver {
            using Real = PetscReal;
            using Int = PetscInt;
        private:

            MPI_Comm comm = MPI_COMM_NULL;

            PartitioningType partioning_type = Naive;
            PartitioningApproach repart_approach = FromScratch;
            ODESolverType odes_type = CVODE_BDF;

            arma::Row<Int> fsp_size;
            arma::Row<Real> fsp_expasion_factors;

            FiniteStateSubset *fsp;
            Vec *p;
            MatrixSet *A;
            FiniteProblemSolver *ode_solver;

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
        public:

            explicit FSPSolver(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type);

            void SetInitFSPSize(arma::Row<Int> &_fsp_size);

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

            FiniteStateSubset &GetStateSubset();

            FSPSolverComponentTiming GetAvgComponentTiming();

            FiniteProblemSolverPerfInfo GetSolverPerfInfo();

            void Solve();

            void Destroy();

            ~FSPSolver();

            friend FiniteStateSubset;
            friend MatrixSet;
            friend FiniteProblemSolver;
        };
    }
}


#endif //PARALLEL_FSP_FSP_H
