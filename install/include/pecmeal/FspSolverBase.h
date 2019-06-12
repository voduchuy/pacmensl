//
// Created by Huy Vo on 5/29/18.
//

#ifndef PECMEAL_FSP_H
#define PECMEAL_FSP_H

#include<algorithm>
#include<cstdlib>
#include<cmath>
#include"Model.h"
#include"DiscreteDistribution.h"
#include"FspMatrixBase.h"
#include"FspMatrixConstrained.h"
#include"OdeSolverBase.h"
#include"StateSetBase.h"
#include"StateSetConstrained.h"
#include"OdeSolver/CvodeFsp.h"
#include"cme_util.h"

namespace pecmeal {
    struct FspSolverComponentTiming {
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

        Model model_;

        std::function<void(PetscReal, Vec, Vec)> tmatvec_;

        arma::Mat<Int> init_states_;
        arma::Col<PetscReal> init_probs_;

        int verbosity_ = 0;
        bool have_custom_constraints_ = false;

        fsp_constr_multi_fn *fsp_constr_funs_;
        arma::Row<int> fsp_bounds_;
        arma::Row<Real> fsp_expasion_factors_;

        // For error checking and expansion parameters
        int CheckFspTolerance_(PetscReal t, Vec p);

        virtual void set_expansion_parameters_() {};
        Real fsp_tol_ = 1.0;
        Real t_final_ = 0.0;

        arma::Row<PetscReal> sinks_;
        arma::Row<int> to_expand_;


        DiscreteDistribution MakeOutputDistribution(PetscReal t, const StateSetBase &state_set, Vec const &p);

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
        NOT_COPYABLE_NOT_MOVABLE(FspSolverBase);

        explicit FspSolverBase(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type);

        void SetConstraintFunctions(fsp_constr_multi_fn *lhs_constr);

        void SetInitialBounds(arma::Row<int> &_fsp_size);

        void SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

        void SetModel(Model &model);

        void SetVerbosity(int verbosity_level);

        void SetInitialDistribution(arma::Mat<Int> &_init_states, arma::Col<PetscReal> &_init_probs);

        void SetLogging(PetscBool logging);

        void SetFromOptions();

        void SetUp();

        Vec &GetP();

        const StateSetBase *GetStateSet();

        FspSolverComponentTiming GetAvgComponentTiming();

        FiniteProblemSolverPerfInfo GetSolverPerfInfo();

        DiscreteDistribution Solve(PetscReal t_final, PetscReal fsp_tol);

        std::vector<DiscreteDistribution> Solve(const arma::Row<PetscReal> &tspan, PetscReal fsp_tol);

        void Destroy();

        ~FspSolverBase();

        friend StateSetBase;
        friend FspMatrixBase;

        friend OdeSolverBase;
    };
}


#endif //PECMEAL_FSP_H
