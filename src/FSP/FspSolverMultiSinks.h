//
// Created by Huy Vo on 5/29/18.
//

#ifndef PACMENSL_FSP_H
#define PACMENSL_FSP_H

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
#include"KrylovFsp.h"
#include"CvodeFsp.h"
#include"Sys.h"

namespace pacmensl {
    struct FspSolverComponentTiming {
        PetscReal StatePartitioningTime;
        PetscReal MatrixGenerationTime;
        PetscReal ODESolveTime;
        PetscReal SolutionScatterTime;
        PetscReal RHSEvalTime;
        PetscReal TotalTime;
    };

    class FspSolverMultiSinks {
        using Real = PetscReal;
        using Int = PetscInt;
    public: NOT_COPYABLE_NOT_MOVABLE( FspSolverMultiSinks );

        explicit FspSolverMultiSinks( MPI_Comm _comm, PartitioningType _part_type = GRAPH,
                                      ODESolverType _solve_type = CVODE_BDF );

        int SetConstraintFunctions( const fsp_constr_multi_fn &lhs_constr );

        int SetInitialBounds( arma::Row< int > &_fsp_size );

        int SetExpansionFactors( arma::Row< PetscReal > &_expansion_factors );

        int SetModel( Model &model );

        int SetVerbosity( int verbosity_level );

        int SetInitialDistribution( const arma::Mat< Int > &_init_states, const arma::Col< PetscReal > &_init_probs );

        int SetLogging( PetscBool logging );

        int SetFromOptions( );

        int SetLoadBalancingMethod( PartitioningType part_type );

        int SetOdesType( ODESolverType odes_type );

        int SetUp( );

        const StateSetBase *GetStateSet( );

        FspSolverComponentTiming GetAvgComponentTiming( );

        FiniteProblemSolverPerfInfo GetSolverPerfInfo( );

        DiscreteDistribution Solve( PetscReal t_final, PetscReal fsp_tol );

        std::vector< DiscreteDistribution > SolveTspan( const std::vector< PetscReal > &tspan, PetscReal fsp_tol );

        int ClearState();

        ~FspSolverMultiSinks( );

    protected:

        MPI_Comm comm_ = nullptr;
        int my_rank_;
        int comm_size_;

        PartitioningType partitioning_type_ = GRAPH;
        PartitioningApproach repart_approach_ = REPARTITION;
        ODESolverType odes_type_ = CVODE_BDF;

        StateSetBase *state_set_ = nullptr;
        Vec *p_ = nullptr;
        FspMatrixBase *A_ = nullptr;
        OdeSolverBase *ode_solver_ = nullptr;
        bool set_up_ = false;

        Model model_;

        std::function< void( PetscReal, Vec, Vec ) > tmatvec_;

        arma::Mat< Int > init_states_;
        arma::Col< PetscReal > init_probs_;

        int verbosity_ = 0;
        bool have_custom_constraints_ = false;

        fsp_constr_multi_fn fsp_constr_funs_;
        arma::Row< int > fsp_bounds_;
        arma::Row< Real > fsp_expasion_factors_;

        // For error checking and expansion parameters
        int CheckFspTolerance_( PetscReal t, Vec p );

        virtual void set_expansion_parameters_( ) {};
        Real fsp_tol_ = 1.0;
        Real t_final_ = 0.0;

        arma::Row< PetscReal > sinks_;
        arma::Row< int > to_expand_;

        DiscreteDistribution Advance_( PetscReal t_final, PetscReal fsp_tol );

        // For logging events using PETSc LogEvent
        PetscBool logging_enabled = PETSC_FALSE;
        PetscLogEvent StateSetPartitioning;
        PetscLogEvent MatrixGeneration;
        PetscLogEvent ODESolve;
        PetscLogEvent SolutionScatter;
        PetscLogEvent RHSEvaluation;
        PetscLogEvent SettingUp;
        PetscLogEvent Solving;
    };
}

#endif //PACMENSL_FSP_H
