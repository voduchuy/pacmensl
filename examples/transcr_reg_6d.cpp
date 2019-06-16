static char help[] = "Solve small CMEs to benchmark intranode performance.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include "FSP/FspSolverBase.h"

#include "models/transcription_regulation_6d_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace six_species_cme;
using namespace cme::parallel;

void output_marginals(MPI_Comm comm, std::string model_name, std::string part_type, std::string part_approach, std::string constraint_type, FspSolverBase& fsp_solver);
void output_performance(MPI_Comm comm, std::string model_name, std::string part_type, std::string part_approach, std::string constraint_type, FspSolverBase& fsp_solver);
void output_time(MPI_Comm comm, std::string model_name, std::string part_type, std::string part_approach, std::string constraint_type, FspSolverBase& fsp_solver);
void petscvec_to_file( MPI_Comm comm, Vec x, const char *filename );

int main( int argc, char *argv[] ) {

    PetscMPIInt ierr, myRank, num_procs;

    ierr = cme::ParaFSP_init( &argc, &argv, help );
    CHKERRQ( ierr );

    // Begin Parallel FSP context
    {
        MPI_Comm comm;
        MPI_Comm_dup( PETSC_COMM_WORLD, &comm );
        MPI_Comm_size( comm, &num_procs );
        PetscPrintf( comm, "\n ================ \n" );

        std::string part_type;
        std::string part_approach;
        // Default problem
        std::string model_name = "transcr_reg_6d";
        PetscReal t_final = 60.00 * 5;
        PetscReal fsp_tol = 1.0e-4;
        arma::Mat< PetscInt > X0 = {2, 6, 0, 2, 0, 0};
        X0 = X0.t( );
        arma::Col< PetscReal > p0 = {1.0};
        arma::Mat< PetscInt > stoich_mat = SM;

        // Default options
        PartitioningType fsp_par_type = Graph;
        PartitioningApproach fsp_repart_approach = Repartition;
        ODESolverType fsp_odes_type = CVODE_BDF;
        PetscBool output_marginal = PETSC_FALSE;
        PetscBool fsp_log_events = PETSC_FALSE;
        // Read options for fsp
        char opt[100];
        PetscBool opt_set;

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
        PetscPrintf( comm, "Solving with %d processors.\n", num_procs );

        ierr = PetscOptionsGetString( NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set );
        CHKERRQ( ierr );
        if ( opt_set ) {
            if ( strcmp( opt, "1" ) == 0 || strcmp( opt, "true" ) == 0 ) {
                fsp_log_events = PETSC_TRUE;
            }
        }

        part_type = part2str( fsp_par_type );
        part_approach = partapproach2str( fsp_repart_approach );
        PetscPrintf( comm, "Partitiniong option %s \n", part2str( fsp_par_type ).c_str( ));
        PetscPrintf( comm, "Repartitoning option %s \n", partapproach2str( fsp_repart_approach ).c_str( ));
        FspSolverBase fsp_solver( PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type );

        // Solve using adaptive default constraints
        fsp_solver.SetInitFSPBounds( rhs_constr_hyperrec );
        fsp_solver.SetExpansionFactors( expansion_factors_hyperrec );
        fsp_solver.SetFSPTolerance( fsp_tol );
        fsp_solver.SetFinalTime( t_final );
        fsp_solver.SetStoichiometry( stoich_mat );
        fsp_solver.SetTimeFunc( t_fun );
        fsp_solver.SetPropensity( propensity );
        fsp_solver.SetInitProbabilities( X0, p0 );
        fsp_solver.SetFromOptions( );
        fsp_solver.SetUp( );
        fsp_solver.Solve( );

        if ( fsp_log_events ) {
            output_time( PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("adaptive_default"), fsp_solver);
            output_performance( PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("adaptive_default"), fsp_solver);
        }
        if ( output_marginal ) {
            output_marginals( PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("adaptive_default"), fsp_solver);
        }

        PetscPrintf( comm, "\n ================ \n" );

        // Solve using fixed default constraints
        FiniteStateSubsetBase* fss = fsp_solver.GetStateSubset();
        arma::Row<int> final_hyperrec_constr = fss->get_shape_bounds( );
        fsp_solver.Destroy();
        fsp_solver.SetInitFSPBounds( final_hyperrec_constr );
        fsp_solver.SetExpansionFactors( expansion_factors_hyperrec );
        fsp_solver.SetFSPTolerance( 1.0e0 );
        fsp_solver.SetFinalTime( t_final );
        fsp_solver.SetStoichiometry( stoich_mat );
        fsp_solver.SetTimeFunc( t_fun );
        fsp_solver.SetPropensity( propensity );
        fsp_solver.SetInitProbabilities( X0, p0 );
        fsp_solver.SetFromOptions( );
        fsp_solver.SetUp( );
        fsp_solver.Solve( );
        if ( fsp_log_events ) {
            output_time( PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("fixed_hyperrec"), fsp_solver);
            output_performance( PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("fixed_hyperrec"), fsp_solver);
        }
        if ( output_marginal ) {
            output_marginals( PETSC_COMM_WORLD, model_name, part_type, part_approach, std::string("fixed_hyperrec"), fsp_solver);
        }
        fsp_solver.Destroy();
    }
    //End Parallel FSP context
    ierr = cme::ParaFSP_finalize( );
    return ierr;
}

void output_marginals(MPI_Comm comm, std::string model_name, std::string part_type, std::string part_approach, std::string constraint_type, FspSolverBase& fsp_solver){
    int myRank, num_procs;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &num_procs);
    /* Compute the marginal distributions */
    Vec P = fsp_solver.GetP( );
    FiniteStateSubsetBase *state_set = fsp_solver.GetStateSubset( );
    std::vector< arma::Col< PetscReal>> marginals( state_set->get_num_species( ));
    for ( PetscInt i{0}; i < marginals.size( ); ++i ) {
        marginals[ i ] = state_set->marginal( P, i );
    }

    MPI_Comm_rank( PETSC_COMM_WORLD, &myRank );
    if ( myRank == 0 ) {
        for ( PetscInt i{0}; i < marginals.size( ); ++i ) {
            std::string filename =
                    model_name + "_marginal_" + std::to_string( i ) + "_" +
                    std::to_string( num_procs ) +
                    "_" +
                    part_type + "_" + part_approach + "_" + constraint_type + ".dat";
            marginals[ i ].save( filename, arma::raw_ascii );
        }
        std::string filename = model_name + "_bounds_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach + "_" + constraint_type + ".dat";
        state_set->get_shape_bounds( ).save(filename, arma::raw_ascii);
    }
}

void output_time(MPI_Comm comm, std::string model_name, std::string part_type, std::string part_approach, std::string constraint_type, FspSolverBase& fsp_solver){
    int myRank, num_procs;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &num_procs);

    FSPSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming( );
    FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo( );
    double solver_time = timings.TotalTime;

    if ( myRank == 0 ) {
        {
            std::string filename =
                    model_name + "_time_" + std::to_string( num_procs ) + "_" + part_type +
                    "_" + part_approach + "_" + constraint_type + ".dat";
            std::ofstream file;
            file.open( filename, std::ios_base::app );
            file << solver_time << "\n";
            file.close( );
        }
    }
}

void output_performance(MPI_Comm comm, std::string model_name, std::string part_type, std::string part_approach, std::string constraint_type, FspSolverBase& fsp_solver){
    int myRank, num_procs;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &num_procs);

    FSPSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming( );
    FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo( );
    double solver_time = timings.TotalTime;
    if ( myRank == 0 ) {
        std::string filename =
                model_name + "_time_breakdown_" + std::to_string( num_procs ) + "_" + part_type +
                "_" + part_approach + "_" + constraint_type + ".dat";
        std::ofstream file;
        file.open( filename );
        file << "Component, Average processor time (sec), Percentage \n";
        file << "Finite State Subset," << std::scientific << std::setprecision( 2 )
             << timings.StatePartitioningTime << ","
             << timings.StatePartitioningTime / solver_time * 100.0
             << "\n"
             << "Matrix Generation," << std::scientific << std::setprecision( 2 )
             << timings.MatrixGenerationTime << ","
             << timings.MatrixGenerationTime / solver_time * 100.0
             << "\n"
             << "Matrix-vector multiplication," << std::scientific << std::setprecision( 2 )
             << timings.RHSEvalTime << "," << timings.RHSEvalTime / solver_time * 100.0 << "\n"
             << "Others," << std::scientific << std::setprecision( 2 )
             << solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime -
                timings.RHSEvalTime << ","
             << ( solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime -
                  timings.RHSEvalTime ) / solver_time * 100.0
             << "\n"
             << "Total," << solver_time << "," << 100.0 << "\n";
        file.close( );

        filename =
                model_name + "_perf_info_" + std::to_string( num_procs ) + "_" + part_type +
                "_" + part_approach + "_" + constraint_type + ".dat";
        file.open( filename );
        file << "Model time, ODEs size, Average processor time (sec) \n";
        for ( auto i{0}; i < perf_info.n_step; ++i ) {
            file << perf_info.model_time[ i ] << "," << perf_info.n_eqs[ i ] << ","
                 << perf_info.cpu_time[ i ]
                 << "\n";
        }
        file.close( );
    }
}

void petscvec_to_file( MPI_Comm comm, Vec x, const char *filename ) {
    PetscViewer viewer;
    PetscViewerCreate( comm, &viewer );
    PetscViewerBinaryOpen( comm, filename, FILE_MODE_WRITE, &viewer );
    VecView( x, viewer );
    PetscViewerDestroy( &viewer );
}
