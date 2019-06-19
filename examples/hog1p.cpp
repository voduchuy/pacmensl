static char help[] = "Advance_ small CMEs to benchmark intranode performance.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include "pacmensl_all.h"


namespace hog1p_cme {
// stoichiometric matrix of the toggle switch model
    arma::Mat< PetscInt > SM{{1, -1, -1, 0, 0, 0,  0,  0,  0},
                             {0, 0,  0,  1, 0, -1, 0,  0,  0},
                             {0, 0,  0,  0, 1, 0,  -1, 0,  0},
                             {0, 0,  0,  0, 0, 1,  0,  -1, 0},
                             {0, 0,  0,  0, 0, 0,  1,  0,  -1},};

// reaction parameters
    const PetscReal k12{1.29}, k23{0.0067}, k34{0.133}, k32{0.027}, k43{0.0381}, k21{1.0e0}, kr21{0.005}, kr31{
            0.45}, kr41{0.025}, kr22{0.0116}, kr32{0.987}, kr42{0.0538}, trans{0.01}, gamma1{0.001}, gamma2{0.0049},
// parameters for the time-dependent factors
            r1{6.9e-5}, r2{7.1e-3}, eta{3.1}, Ahog{9.3e09}, Mhog{6.4e-4};

// propensity function
    PetscReal propensity( const PetscInt *X, const PetscInt k ) {
        switch ( k ) {
            case 0:
                return k12 * double( X[ 0 ] == 0 ) + k23 * double( X[ 0 ] == 1 ) + k34 * double( X[ 0 ] == 2 );
            case 1:
                return k32 * double( X[ 0 ] == 2 ) + k43 * double( X[ 0 ] == 3 );
            case 2:
                return k21 * double( X[ 0 ] == 1 );
            case 3:
                return kr21 * double( X[ 0 ] == 1 ) + kr31 * double( X[ 0 ] == 2 ) + kr41 * double( X[ 0 ] == 3 );
            case 4:
                return kr22 * double( X[ 0 ] == 1 ) + kr32 * double( X[ 0 ] == 2 ) + kr42 * double( X[ 0 ] == 3 );
            case 5:
                return trans * double( X[ 1 ] );
            case 6:
                return trans * double( X[ 2 ] );
            case 7:
                return gamma1 * double( X[ 3 ] );
            case 8:
                return gamma2 * double( X[ 4 ] );
            default:
                return 0.0;
        }
    }

// function to compute the time-dependent coefficients of the propensity functions
    arma::Row< double > t_fun( double t ) {
        arma::Row< double > u( 9, arma::fill::ones );

        double h1 = ( 1.0 - exp( -r1 * t )) * exp( -r2 * t );

        double hog1p = pow( h1 / ( 1.0 + h1 / Mhog ), eta ) * Ahog;

        u( 2 ) = std::max( 0.0, 3200.0 - 7710.0 * ( hog1p ));

        return u;
    }

    // Function to constraint the shape of the FSP
    void lhs_constr( PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,
                     void *args ) {

        for ( int j{0}; j < num_states; ++j ) {
            for ( int i{0}; i < 5; ++i ) {
                vals[ num_constrs * j + i ] = ( states[ num_species * j + i ] );
            }
            vals[ num_constrs * j + 5 ] = (( states[ num_species * j + 1 ] ) + ( states[ num_species * j + 3 ] ));
            vals[ num_constrs * j + 6 ] = (( states[ num_species * j + 2 ] ) + ( states[ num_species * j + 4 ] ));
        }
    }

    arma::Row< int > rhs_constr_hyperrec{3, 10, 10, 10, 10};
    arma::Row< double > expansion_factors_hyperrec{0.0, 0.25, 0.25, 0.25, 0.25};
    arma::Row< int > rhs_constr{3, 10, 10, 10, 10, 10, 10};
    arma::Row< double > expansion_factors{0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
}

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace hog1p_cme;
using namespace pacmensl;

void output_marginals( MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                       PartitioningApproach fsp_repart_approach, std::string constraint_type,
                       DiscreteDistribution &solution, arma::Row< int > constraints );

void output_time( MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                  PartitioningApproach fsp_repart_approach, std::string constraint_type, FspSolverBase &fsp_solver );

void output_performance( MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                         PartitioningApproach fsp_repart_approach, std::string constraint_type,
                         FspSolverBase &fsp_solver );

int ParseOptions( MPI_Comm comm, PartitioningType &fsp_par_type, PartitioningApproach &fsp_repart_approach,
                  PetscBool &output_marginal, PetscBool &fsp_log_events );

int main( int argc, char *argv[] ) {

    PetscMPIInt ierr, myRank, num_procs;

    ierr = pacmensl::PACMENSLInit( &argc, &argv, help );
    CHKERRQ( ierr );

    MPI_Comm comm;
    MPI_Comm_dup( PETSC_COMM_WORLD, &comm );
    MPI_Comm_size( comm, &num_procs );
    PetscPrintf( comm, "\n ================ \n" );
    PetscPrintf( comm, "Solving with %d processors.\n", num_procs );

    std::string model_name = "hog1p";
    Model hog1p_model( hog1p_cme::SM, hog1p_cme::t_fun, hog1p_cme::propensity );

    PetscReal t_final = 60 * 5.0;
    PetscReal fsp_tol = 1.0e-4;
    arma::Mat< PetscInt > X0 = {0, 0, 0, 0, 0};
    X0 = X0.t( );
    arma::Col< PetscReal > p0 = {1.0};


    // Default options
    PartitioningType fsp_par_type = GRAPH;
    PartitioningApproach fsp_repart_approach = REPARTITION;
    ODESolverType fsp_odes_type = CVODE_BDF;
    PetscBool output_marginal = PETSC_FALSE;
    PetscBool fsp_log_events = PETSC_FALSE;

    ierr = ParseOptions( comm, fsp_par_type, fsp_repart_approach, output_marginal, fsp_log_events );
    CHKERRQ( ierr );


    FspSolverBase fsp_solver( comm, fsp_par_type, CVODE_BDF );
    fsp_solver.SetFromOptions( );
    DiscreteDistribution solution;

    // Advance_ using adaptive custom constraints
    fsp_solver.SetConstraintFunctions( &hog1p_cme::lhs_constr );
    fsp_solver.SetModel( hog1p_model );
    fsp_solver.SetInitialBounds( hog1p_cme::rhs_constr );
    fsp_solver.SetExpansionFactors( hog1p_cme::expansion_factors );
    fsp_solver.SetInitialDistribution( X0, p0 );
    fsp_solver.SetUp( );
    solution = fsp_solver.Solve( t_final, fsp_tol );
    const StateSetConstrained *fss = ( StateSetConstrained * ) fsp_solver.GetStateSet( );
    arma::Row< int > final_custom_constr = fss->GetShapeBounds( );
    if ( fsp_log_events ) {
        output_time( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string( "adaptive_custom" ),
                     fsp_solver );
        output_performance( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                            std::string( "adaptive_custom" ), fsp_solver );
    }
    if ( output_marginal ) {
        output_marginals( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                          std::string( "adaptive_custom" ), solution, final_custom_constr );
    }

    // Advance_ using fixed custom constraints
    fsp_solver.Destroy( );
    fsp_solver.SetConstraintFunctions( &hog1p_cme::lhs_constr );
    fsp_solver.SetInitialBounds( final_custom_constr );
    fsp_solver.SetUp( );
    solution = fsp_solver.Solve( t_final, fsp_tol );
    if ( fsp_log_events ) {
        output_time( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string( "fixed_custom" ),
                     fsp_solver );
        output_performance( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                            std::string( "fixed_custom" ), fsp_solver );
    }
    if ( output_marginal ) {
        output_marginals( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                          std::string( "fixed_custom" ), solution, final_custom_constr );
    }

    // Advance_ using adaptive default constraints
    fsp_solver.Destroy( );
    fsp_solver.SetInitialBounds( rhs_constr_hyperrec );
    fsp_solver.SetExpansionFactors( expansion_factors_hyperrec );
    fsp_solver.SetFromOptions( );
    fsp_solver.SetUp( );
    solution = fsp_solver.Solve( t_final, fsp_tol );
    fss = ( StateSetConstrained * ) fsp_solver.GetStateSet( );
    arma::Row< int > final_hyperrec_constr = fss->GetShapeBounds( );
    if ( fsp_log_events ) {
        output_time( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string( "adaptive_default" ),
                     fsp_solver );
        output_performance( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                            std::string( "adaptive_default" ), fsp_solver );
    }
    if ( output_marginal ) {
        output_marginals( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                          std::string( "adaptive_default" ), solution, final_hyperrec_constr );
    }
    fsp_solver.Destroy( );
    PetscPrintf( comm, "\n ================ \n" );

    // Advance_ using fixed default constraints
    fsp_solver.SetConstraintFunctions( &hog1p_cme::lhs_constr );
    fsp_solver.SetInitialBounds( final_hyperrec_constr );
    fsp_solver.SetUp( );
    solution = fsp_solver.Solve( t_final, fsp_tol );
    if ( fsp_log_events ) {
        output_time( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string( "fixed_hyperrec" ),
                     fsp_solver );
        output_performance( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                            std::string( "fixed_hyperrec" ), fsp_solver );
    }
    if ( output_marginal ) {
        output_marginals( PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                          std::string( "fixed_hyperrec" ), solution, final_hyperrec_constr );
    }
    fsp_solver.Destroy( );

    ierr = pacmensl::PACMENSLFinalize( );
    return ierr;
}

int ParseOptions( MPI_Comm comm, PartitioningType &fsp_par_type, PartitioningApproach &fsp_repart_approach,
                  PetscBool &output_marginal, PetscBool &fsp_log_events ) {
    std::string part_type;
    std::string part_approach;
    part_type = part2str( fsp_par_type );
    part_approach = partapproach2str( fsp_repart_approach );

    // Read options for fsp
    char opt[100];
    PetscBool opt_set;
    int ierr;
    ierr = PetscOptionsGetString( NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set );
    CHKERRQ( ierr );
    if ( opt_set ) {
        fsp_par_type = str2part( std::string( opt ));
    }

    ierr = PetscOptionsGetString( NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set );
    CHKERRQ( ierr );
    if ( opt_set ) {
        fsp_repart_approach = str2partapproach( std::string( opt ));
    }

    ierr = PetscOptionsGetString( NULL, PETSC_NULL, "-fsp_output_marginal", opt, 100, &opt_set );
    CHKERRQ( ierr );
    if ( opt_set ) {
        if ( strcmp( opt, "1" ) == 0 || strcmp( opt, "true" ) == 0 ) {
            output_marginal = PETSC_TRUE;
        }
    }

    ierr = PetscOptionsGetString( NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set );
    CHKERRQ( ierr );
    if ( opt_set ) {
        if ( strcmp( opt, "1" ) == 0 || strcmp( opt, "true" ) == 0 ) {
            fsp_log_events = PETSC_TRUE;
        }
    }
    PetscPrintf( comm, "Partitiniong option %s \n", part2str( fsp_par_type ).c_str( ));
    PetscPrintf( comm, "Repartitoning option %s \n", partapproach2str( fsp_repart_approach ).c_str( ));
    return 0;
}

void output_performance( MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                         PartitioningApproach fsp_repart_approach, std::string constraint_type,
                         FspSolverBase &fsp_solver ) {
    int myRank, num_procs;
    MPI_Comm_rank( comm, &myRank );
    MPI_Comm_size( comm, &num_procs );

    std::string part_type;
    std::string part_approach;
    part_type = part2str( fsp_par_type );
    part_approach = partapproach2str( fsp_repart_approach );

    FspSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming( );
    FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo( );
    double solver_time = timings.TotalTime;
    if ( myRank == 0 ) {
        std::string filename =
                model_name + "_time_breakdown_" + std::to_string( num_procs ) + "_" + part_type + "_" + part_approach +
                "_" + constraint_type + ".dat";
        std::ofstream file;
        file.open( filename );
        file << "Component, Average processor time (sec), Percentage \n";
        file << "Finite State Subset," << std::scientific << std::setprecision( 2 ) << timings.StatePartitioningTime
             << "," << timings.StatePartitioningTime / solver_time * 100.0 << "\n" << "Matrix Generation,"
             << std::scientific << std::setprecision( 2 ) << timings.MatrixGenerationTime << ","
             << timings.MatrixGenerationTime / solver_time * 100.0 << "\n" << "Matrix-vector multiplication,"
             << std::scientific << std::setprecision( 2 ) << timings.RHSEvalTime << ","
             << timings.RHSEvalTime / solver_time * 100.0 << "\n" << "Others," << std::scientific
             << std::setprecision( 2 )
             << solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime - timings.RHSEvalTime << ","
             << ( solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime - timings.RHSEvalTime ) /
                solver_time * 100.0 << "\n" << "Total," << solver_time << "," << 100.0 << "\n";
        file.close( );

        filename =
                model_name + "_perf_info_" + std::to_string( num_procs ) + "_" + part_type + "_" + part_approach + "_" +
                constraint_type + ".dat";
        file.open( filename );
        file << "Model time, ODEs size, Average processor time (sec) \n";
        for ( auto i{0}; i < perf_info.n_step; ++i ) {
            file << perf_info.model_time[ i ] << "," << perf_info.n_eqs[ i ] << "," << perf_info.cpu_time[ i ] << "\n";
        }
        file.close( );
    }
}

void output_time( MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                  PartitioningApproach fsp_repart_approach, std::string constraint_type, FspSolverBase &fsp_solver ) {
    int myRank, num_procs;
    MPI_Comm_rank( comm, &myRank );
    MPI_Comm_size( comm, &num_procs );

    std::string part_type;
    std::string part_approach;
    part_type = part2str( fsp_par_type );
    part_approach = partapproach2str( fsp_repart_approach );

    FspSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming( );
    FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo( );
    double solver_time = timings.TotalTime;

    if ( myRank == 0 ) {
        {
            std::string filename =
                    model_name + "_time_" + std::to_string( num_procs ) + "_" + part_type + "_" + part_approach + "_" +
                    constraint_type + ".dat";
            std::ofstream file;
            file.open( filename, std::ios_base::app );
            file << solver_time << "\n";
            file.close( );
        }
    }
}

void output_marginals( MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                       PartitioningApproach fsp_repart_approach, std::string constraint_type,
                       DiscreteDistribution &solution, arma::Row< int > constraints ) {
    int myRank, num_procs;
    MPI_Comm_rank( comm, &myRank );
    MPI_Comm_size( comm, &num_procs );

    std::string part_type;
    std::string part_approach;
    part_type = part2str( fsp_par_type );
    part_approach = partapproach2str( fsp_repart_approach );

    /* Compute the marginal distributions */
    std::vector< arma::Col< PetscReal>> marginals( solution.states.n_rows );
    for ( PetscInt i{0}; i < marginals.size( ); ++i ) {
        marginals[ i ] = Compute1DMarginal( solution, i );
    }

    MPI_Comm_rank( PETSC_COMM_WORLD, &myRank );
    if ( myRank == 0 ) {
        for ( PetscInt i{0}; i < marginals.size( ); ++i ) {
            std::string filename =
                    model_name + "_marginal_" + std::to_string( i ) + "_" + std::to_string( num_procs ) + "_" +
                    part_type + "_" + part_approach + "_" + constraint_type + ".dat";
            marginals[ i ].save( filename, arma::raw_ascii );
        }
    }

    std::string filename =
            model_name + "_constraint_bounds_" + std::to_string( num_procs ) + "_" + part_type + "_" + part_approach +
            "_" + constraint_type + ".dat";
    constraints.save( filename, arma::raw_ascii );
}
