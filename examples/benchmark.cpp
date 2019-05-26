static char help[] = "Solve small CMEs to benchmark intranode performance.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <util/cme_util.h>
#include <armadillo>
#include <cmath>
#include "FSP/FSPSolver.h"

#include "models/repressilator_model.h"
#include "models/toggle_model.h"
#include "models/hog1p_5d_model.h"
#include "models/transcription_regulation_6d_model.h"
#include "models/hog1p_3d_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file( MPI_Comm comm, Vec x, const char *filename );

using namespace hog1p_cme;
using namespace cme::parallel;

int main( int argc, char *argv[] ) {

    PetscMPIInt ierr, myRank, num_procs;

    ierr = cme::ParaFSP_init( &argc, &argv, help );
    CHKERRQ( ierr );

    PetscPreLoadBegin( PETSC_TRUE, "Stage 0" );

            // Begin Parallel FSP context
            {
                MPI_Comm comm;
                MPI_Comm_dup( PETSC_COMM_WORLD, &comm );
                MPI_Comm_size( comm, &num_procs );
                PetscPrintf( comm, "\n ================ \n" );

                std::string part_type;
                std::string part_approach;
                // Default problem
                std::string model_name = "hog1p";
                fsp_constr_multi_fn *FSPConstraintFuns = hog1p_cme::lhs_constr;
                Row< int > FSPBounds = hog1p_cme::rhs_constr; // Size of the FSP
                arma::Row< PetscReal > expansion_factors = hog1p_cme::expansion_factors;
                PetscReal t_final = 60.00 * 5;
                PetscReal fsp_tol = 1.0e-4;
                arma::Mat< PetscInt > X0 = {0, 0, 0, 0, 0};
                X0 = X0.t( );
                arma::Col< PetscReal > p0 = {1.0};
                arma::Mat< PetscInt > stoich_mat = hog1p_cme::SM;
                TcoefFun t_fun = hog1p_cme::t_fun;
                PropFun propensity = hog1p_cme::propensity;

                // Default options
                PartitioningType fsp_par_type = Graph;
                PartitioningApproach fsp_repart_approach = FromScratch;
                ODESolverType fsp_odes_type = CVODE_BDF;
                PetscBool output_marginal = PETSC_FALSE;
                PetscBool fsp_log_events = PETSC_FALSE;
                // Read options for fsp
                char opt[100];
                PetscBool opt_set;

                ierr = PetscOptionsGetString( NULL, PETSC_NULL, "-fsp_model", opt, 100, &opt_set );
                CHKERRQ( ierr );
                if ( opt_set ) {
                    if ( strcmp( opt, "transcr_reg_6d" ) == 0 ) {
                        model_name = "transcr_reg_6d";
                        FSPConstraintFuns = six_species_cme::lhs_constr;
                        FSPBounds = six_species_cme::rhs_constr; // Size of the FSP
                        expansion_factors = six_species_cme::expansion_factors;
                        t_final = 300.0;
                        fsp_tol = 1.0e-4;
                        X0 = {2, 6, 0, 2, 0, 0};
                        X0 = X0.t( );
                        p0 = {1.0};
                        stoich_mat = six_species_cme::SM;
                        t_fun = six_species_cme::t_fun;
                        propensity = six_species_cme::propensity;
                        PetscPrintf( PETSC_COMM_WORLD, "Problem: Transcription regulation with 6 species.\n" );
                    } else if ( strcmp( opt, "hog3d" ) == 0 ) {
                        model_name = "hog3d";
                        FSPConstraintFuns = hog3d_cme::lhs_constr;
                        FSPBounds = hog3d_cme::rhs_constr; // Size of the FSP
                        expansion_factors = hog3d_cme::expansion_factors;
                        t_final = 60.00 * 15;
                        fsp_tol = 1.0e-4;
                        X0 = {0, 0, 0};
                        X0 = X0.t( );
                        p0 = {1.0};
                        stoich_mat = hog3d_cme::SM;
                        t_fun = hog3d_cme::t_fun;
                        propensity = hog3d_cme::propensity;
                        PetscPrintf( PETSC_COMM_WORLD, "Problem: Hog1p with 3 species.\n" );
                    } else if ( strcmp (opt, "repressilator") == 0){
                        model_name = "repressilator";
                        FSPConstraintFuns = repressilator_cme::lhs_constr;
                        FSPBounds = repressilator_cme::rhs_constr; // Size of the FSP
                        expansion_factors = repressilator_cme::expansion_factors;
                        t_final = 10.0;
                        fsp_tol = 1.0e-4;
                        X0 = {20, 0, 0};
                        X0 = X0.t( );
                        p0 = {1.0};
                        stoich_mat = repressilator_cme::SM;
                        t_fun = repressilator_cme::t_fun;
                        propensity = repressilator_cme::propensity;
                        PetscPrintf( PETSC_COMM_WORLD, "Problem: Repressilator with 3 species.\n" );
                    } else {
                        PetscPrintf( PETSC_COMM_WORLD, "Problem: Hog1p with 5 species.\n" );
                    }
                }


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

                PetscReal tic, tic1, solver_time, total_time;


                tic = MPI_Wtime( );
                FSPSolver fsp_solver( PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type );
                fsp_solver.SetInitFSPBounds( FSPBounds );
                fsp_solver.SetFSPConstraintFunctions( FSPConstraintFuns );
                fsp_solver.SetExpansionFactors( expansion_factors );
                fsp_solver.SetFSPTolerance( fsp_tol );
                if ( PetscPreLoadIt == 0 ) {
                    fsp_solver.SetFinalTime( 1.0e-6 );
                } else {
                    fsp_solver.SetFinalTime( t_final );
                }
                fsp_solver.SetStoichiometry( stoich_mat );
                fsp_solver.SetTimeFunc( t_fun );
                fsp_solver.SetPropensity( propensity );
                fsp_solver.SetInitProbabilities( X0, p0 );
                fsp_solver.SetFromOptions( );

                tic1 = MPI_Wtime( );

                fsp_solver.SetUp( );
                fsp_solver.Solve( );

                solver_time = MPI_Wtime( ) - tic1;
                total_time = MPI_Wtime( ) - tic;

                PetscPrintf( comm, "Total time (including setting up) = %.2e \n", total_time );
                PetscPrintf( comm, "Solving time = %.2e \n", solver_time );

                if ( PetscPreLoadIt == 1 ) {
                    MPI_Comm_rank( PETSC_COMM_WORLD, &myRank );
                    if ( myRank == 0 ) {
                        {
                            std::string filename =
                                    model_name + "_time_" + std::to_string( num_procs ) + "_" + part_type +
                                    "_" + part_approach + ".dat";
                            std::ofstream file;
                            file.open( filename, std::ios_base::app );
                            file << solver_time << "\n";
                            file.close( );
                        }
                    }

                    if ( fsp_log_events ) {
                        FSPSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming( );
                        FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo( );
                        if ( myRank == 0 ) {
                            std::string filename =
                                    model_name + "_time_breakdown_" + std::to_string( num_procs ) + "_" + part_type +
                                    "_" + part_approach + ".dat";
                            std::ofstream file;
                            file.open( filename );
                            file << "Component, Average processor time (sec), Percentage \n";
                            file << "State Partitioning," << std::scientific << std::setprecision( 2 )
                                 << timings.StatePartitioningTime << ","
                                 << timings.StatePartitioningTime / solver_time * 100.0
                                 << "\n"
                                 << "FSP Matrices Generation," << std::scientific << std::setprecision( 2 )
                                 << timings.MatrixGenerationTime << ","
                                 << timings.MatrixGenerationTime / solver_time * 100.0
                                 << "\n"
                                 << "Solving truncated ODEs," << std::scientific << std::setprecision( 2 )
                                 << timings.ODESolveTime
                                 << "," << timings.ODESolveTime / solver_time * 100.0 << "\n"
                                 << "FSP Solution scattering," << std::scientific << std::setprecision( 2 )
                                 << timings.SolutionScatterTime << ","
                                 << timings.SolutionScatterTime / solver_time * 100.0
                                 << "\n"
                                 << "Time-dependent Matrix action," << std::scientific << std::setprecision( 2 )
                                 << timings.RHSEvalTime << "," << timings.RHSEvalTime / solver_time * 100.0 << "\n"
                                 << "Total," << solver_time << "," << 100.0 << "\n";
                            file.close( );

                            filename =
                                    model_name + "_perf_info_" + std::to_string( num_procs ) + "_" + part_type +
                                    "_" + part_approach + ".dat";
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

                    if ( output_marginal ) {
                        /* Compute the marginal distributions */
                        Vec P = fsp_solver.GetP( );
                        FiniteStateSubset &state_set = fsp_solver.GetStateSubset( );
                        std::vector< arma::Col< PetscReal>> marginals( state_set.GetNumSpecies( ));
                        for ( PetscInt i{0}; i < marginals.size( ); ++i ) {
                            marginals[ i ] = state_set.marginal( P, i );
                        }

                        MPI_Comm_rank( PETSC_COMM_WORLD, &myRank );
                        if ( myRank == 0 ) {
                            for ( PetscInt i{0}; i < marginals.size( ); ++i ) {
                                std::string filename =
                                        model_name + "_marginal_" + std::to_string( i ) + "_" +
                                        std::to_string( num_procs ) +
                                        "_" +
                                        part_type + "_" + part_approach + ".dat";
                                marginals[ i ].save( filename, arma::raw_ascii );
                            }
                        }
                    }
                }

                PetscPrintf( comm, "\n ================ \n" );
            }
    PetscPreLoadEnd( );
    //End Parallel FSP context
    ierr = cme::ParaFSP_finalize( );
    return ierr;
}


void petscvec_to_file( MPI_Comm comm, Vec x, const char *filename ) {
    PetscViewer viewer;
    PetscViewerCreate( comm, &viewer );
    PetscViewerBinaryOpen( comm, filename, FILE_MODE_WRITE, &viewer );
    VecView( x, viewer );
    PetscViewerDestroy( &viewer );
}
