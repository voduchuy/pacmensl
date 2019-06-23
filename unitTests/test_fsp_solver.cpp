//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include <gtest/gtest.h>
#include "pacmensl_all.h"
#include "FspSolverMultiSinks.h"
#include "my_test_env.h"


namespace toggle_cme {
/* Stoichiometric matrix of the toggle switch model */
    arma::Mat< PetscInt > SM{{1, 1, -1, 0, 0, 0},
                             {0, 0, 0,  1, 1, -1}};

    const int nReaction = 6;

/* Parameters for the propensity functions */
    const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3.0e0}, nxy{2.1e0}, kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{
            1.6e-2}, dy{3.8e-4};

    // Function to constraint the shape of the FSP
    void lhs_constr( PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,
                     void *args ) {

        for ( int i{0}; i < num_states; ++i ) {
            vals[ i * num_constrs ] = states[ num_species * i ];
            vals[ i * num_constrs + 1 ] = states[ num_species * i + 1 ];
            vals[ i * num_constrs + 2 ] = states[ num_species * i ] * states[ num_species * i + 1 ];
        }
    }

    arma::Row< int > rhs_constr{200, 200, 2000};
    arma::Row< double > expansion_factors{0.2, 0.2, 0.2};

// propensity function for toggle
    int propensity( const int reaction, const int num_species, const int num_states, const PetscInt *X, double *outputs,
                    void *args ) {
        int (*X_view)[2] = ( int ( * )[2] ) X;
        switch ( reaction ) {
            case 0:
                for ( int i{0}; i < num_states; ++i ) { outputs[ i ] = 1.0; }
                break;
            case 1:
                for ( int i{0}; i < num_states; ++i ) {
                    outputs[ i ] = 1.0 / ( 1.0 + ayx * pow( PetscReal( X_view[ i ][ 1 ] ), nyx ));
                }
                break;
            case 2:
                for ( int i{0}; i < num_states; ++i ) { outputs[ i ] = PetscReal( X_view[ i ][ 0 ] ); }
                break;
            case 3:
                for ( int i{0}; i < num_states; ++i ) { outputs[ i ] = 1.0; }
                break;
            case 4:
                for ( int i{0}; i < num_states; ++i ) {
                    outputs[ i ] = 1.0 / ( 1.0 + axy * pow( PetscReal( X_view[ i ][ 0 ] ), nxy ));
                }
                break;
            case 5:
                for ( int i{0}; i < num_states; ++i ) { outputs[ i ] = PetscReal( X_view[ i ][ 1 ] ); }
                break;
            default:
                return -1;
        }
        return 0;
    }

    int t_fun( PetscReal t, int n_coefs, double *outputs, void *args ) {
        outputs[ 0 ] = kx0;
        outputs[ 1 ] = kx;
        outputs[ 2 ] = dx;
        outputs[ 3 ] = ky0;
        outputs[ 4 ] = ky;
        outputs[ 5 ] = dy;
        return 0;
    }
}

using namespace pacmensl;

class FspTest : public ::testing::Test {
protected:
    void SetUp( ) override {
        t_final = 100.0;
        fsp_tol = 1.0e-8;
        X0 = X0.t( );
        toggle_model = Model( toggle_cme::SM, toggle_cme::t_fun, nullptr, toggle_cme::propensity, nullptr );
    }

    void TearDown( ) override {

    }

    PetscReal t_final, fsp_tol;
    arma::Mat< PetscInt > X0{0, 0};
    arma::Col< PetscReal > p0 = {1.0};

    Model toggle_model;
    arma::Row< int > fsp_size = {50, 50};
    arma::Row< PetscReal > expansion_factors = {0.25, 0.25};
};

TEST_F( FspTest, test_wrong_call_sequence_detection){
    int ierr;
    FspSolverMultiSinks fsp( PETSC_COMM_WORLD );
    ierr = fsp.SetUp();
    ASSERT_EQ(ierr, -1);
}

TEST_F( FspTest, toggle_compare_ode_methods ) {
    PetscInt ierr;
    PetscReal stmp;
    DiscreteDistribution p_final_bdf, p_final_krylov;
    std::vector< DiscreteDistribution > p_snapshots_bdf, p_snapshots_krylov;
    std::vector< PetscReal > tspan;
    Vec q;

    // Get processor rank and number of processors
    PetscMPIInt rank, num_procs;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    MPI_Comm_size( PETSC_COMM_WORLD, &num_procs );

    tspan = arma::conv_to< std::vector< PetscReal>>::from( arma::linspace< arma::Row< PetscReal>>( 0.0, t_final, 3 ));

    FspSolverMultiSinks fsp( PETSC_COMM_WORLD );

    fsp.SetModel( toggle_model );
    fsp.SetInitialBounds( fsp_size );
    fsp.SetExpansionFactors( expansion_factors );
    fsp.SetVerbosity( 0 );
    fsp.SetInitialDistribution( X0, p0 );

    fsp.SetOdesType( CVODE_BDF );
    fsp.SetUp( );
    p_final_bdf = fsp.Solve( t_final, fsp_tol );
    p_snapshots_bdf = fsp.SolveTspan( tspan, fsp_tol );
  fsp.ClearState();

    fsp.SetOdesType( KRYLOV );
    fsp.SetUp( );
    p_final_krylov = fsp.Solve( t_final, fsp_tol );
    p_snapshots_krylov = fsp.SolveTspan( tspan, fsp_tol );

    // Choice of ODE solvers must not affect the state space size
    int bdf_vec_size, krylov_vec_size;
    ierr = VecGetSize( p_final_bdf.p_, &bdf_vec_size );
    ASSERT_FALSE( ierr );
    ierr = VecGetSize( p_final_krylov.p_, &krylov_vec_size );
    ASSERT_FALSE( ierr );
    ASSERT_EQ( bdf_vec_size, krylov_vec_size );

    ierr = VecDuplicate( p_final_bdf.p_, &q );
    ASSERT_FALSE( ierr );

    ierr = VecCopy( p_final_bdf.p_, q );
    ASSERT_FALSE( ierr );

    ierr = VecAXPY( q, -1.0, p_final_krylov.p_ );
    ASSERT_FALSE( ierr );
    ierr = VecNorm( q, NORM_1, &stmp );
    ASSERT_FALSE( ierr );
    PetscPrintf( PETSC_COMM_WORLD, "Final solution Krylov - BDF = %.2e \n", stmp );

    for ( int i{0}; i < tspan.size( ); ++i ) {
        ierr = VecCopy( p_snapshots_bdf[ i ].p_, q );
        ASSERT_FALSE( ierr );
        ierr = VecAXPY( q, -1.0, p_snapshots_krylov[ i ].p_ );
        ASSERT_FALSE( ierr );
        ierr = VecNorm( q, NORM_1, &stmp );
        ASSERT_FALSE( ierr );
        PetscPrintf( PETSC_COMM_WORLD, "Final solution at t = %.2e Krylov - BDF = %.2e \n", tspan.at( i ), stmp );

        ierr = VecSum( p_final_bdf.p_, &stmp );
        ASSERT_FALSE( ierr );
        PetscPrintf( PETSC_COMM_WORLD, "Sum(p_final_bdf) = %.2e \n", stmp );

        ierr = VecSum( p_final_krylov.p_, &stmp );
        ASSERT_FALSE( ierr );
        PetscPrintf( PETSC_COMM_WORLD, "Sum(p_final_krylov) = %.2e \n", stmp );
    }

    ierr = VecDestroy( &q );
    ASSERT_FALSE( ierr );
}