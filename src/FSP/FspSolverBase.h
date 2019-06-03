//
// Created by Huy Vo on 5/29/18.
//

#ifndef PARALLEL_FSP_FSP_H
#define PARALLEL_FSP_FSP_H

#include<algorithm>
#include<cstdlib>
#include<cmath>
#include"Matrix/FspMatrixBase.h"
#include"Matrix/FspMatrixConstrained.h"
#include"OdeSolver/OdeSolverBase.h"
#include"FSS/StateSetBase.h"
#include"FSS/StateSetConstrained.h"
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

        struct FspSolution{
            MPI_Comm comm;
            double t;
            arma::Mat<int> states;
            Vec p;
        };

        class FspSolverBase {
            using Real = PetscReal;
            using Int = PetscInt;
        private:

            MPI_Comm comm_ = MPI_COMM_NULL;
            int my_rank_;
            int comm_size_;

            PartitioningType partitioning_type_ = Graph;
            PartitioningApproach repart_approach_ = Repartition;
            ODESolverType odes_type = CVODE_BDF;

            StateSetBase *state_set_;
            Vec *p_;
            FspMatrixBase *A_;
            OdeSolverBase *ode_solver_;

            Real t_final = 0.0;
            Real fsp_tol = 0.0;

            arma::Mat<Int> stoich_mat;
            PropFun propensity_;
            TcoefFun t_fun_;

            std::function<void(PetscReal, Vec, Vec)> tmatvec_;
            arma::Mat<Int> init_states_;

            arma::Col<PetscReal> init_probs_;
            int verbosity = 0;

            bool have_custom_constraints_ = false;
            fsp_constr_multi_fn *fsp_constr_funs_;
            arma::Row<int> fsp_bounds_;
            arma::Row<Real> fsp_expasion_factors_;

            // For error checking and expansion parameters
            int check_fsp_tolerance_(PetscReal t, Vec p);

            arma::Row<PetscReal> sinks_;
            arma::Row<int> to_expand_;

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
