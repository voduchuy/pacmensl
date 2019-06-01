//
// Created by Huy Vo on 12/6/18.
//
#include <FPSolver/cvode_interface/CVODEFSP.h>

#include "FPSolver/OdeSolverBase.h"
#include "CVODEFSP.h"

namespace cme {
    namespace parallel {

        CVODEFSP::CVODEFSP( MPI_Comm _comm, int lmm ) : OdeSolverBase( _comm ) {
            cvode_mem = CVodeCreate( lmm);
            if ( cvode_mem == nullptr ) {
                throw std::runtime_error( "CVODE failed to initialize memory.\n" );
            }
            solver_type = CVODE_BDF;
        }

        PetscInt CVODEFSP::Solve( ) {
            // Make sure the necessary data has been set
            assert( solution != nullptr );
            assert( rhs );
            assert( fsp != nullptr );

            PetscInt petsc_err;

            // N_Vector wrapper for the solution
            solution_wrapper = N_VMake_Petsc( *solution );

            // Copy solution to the temporary solution
            solution_tmp = N_VClone( solution_wrapper );
            Vec solution_tmp_dat = N_VGetVector_Petsc( solution_tmp );
            petsc_err = VecSetUp( solution_tmp_dat );
            CHKERRABORT( comm, petsc_err );
            petsc_err = VecCopy( *solution, solution_tmp_dat );
            CHKERRABORT( comm, petsc_err );

            // Set CVODE starting time to the current timepoint
            t_now_tmp = t_now;

            // Initialize cvode
            cvode_stat = CVodeInit( cvode_mem, &cvode_rhs, t_now_tmp, solution_tmp );
            CVODECHKERR( comm, cvode_stat );
            cvode_stat = CVodeSetUserData( cvode_mem, ( void * ) this );
            CVODECHKERR( comm, cvode_stat );
            cvode_stat = CVodeSStolerances( cvode_mem, rel_tol, abs_tol );
            CVODECHKERR( comm, cvode_stat );
            cvode_stat = CVodeSetMaxNumSteps( cvode_mem, 10000 );
            CVODECHKERR( comm, cvode_stat );
            cvode_stat = CVodeSetMaxConvFails( cvode_mem, 10000 );
            CVODECHKERR( comm, cvode_stat );
            cvode_stat = CVodeSetMaxNonlinIters( cvode_mem, 10000 );
            CVODECHKERR( comm, cvode_stat );

            // Create the linear solver without preconditioning
            linear_solver = SUNSPBCGS( solution_tmp, PREC_NONE, 0 );
            cvode_stat = CVSpilsSetLinearSolver( cvode_mem, linear_solver );
            CVODECHKERR( comm, cvode_stat );
            cvode_stat = CVSpilsSetJacTimes( cvode_mem, NULL, &cvode_jac );
            CVODECHKERR( comm, cvode_stat );

            // Advance the temporary solution until either reaching final time or FSP error exceeding tolerance
            arma::Row< PetscReal > sink_values( fsp->get_num_constraints( ));
            arma::Row< PetscReal > sink_values_old( fsp->get_num_constraints( ));
            PetscBool fsp_accept;
            expand_sink.fill( 0 );
            sink_values_old = fsp->SinkStatesReduce( solution_tmp_dat );
            for ( int j = 0; j < sink_values_old.n_elem; ++j ) {
                sink_values_old(j) = round2digit(sink_values_old(j));
            }
            while ( t_now < t_final ) {
                cvode_stat = CVode( cvode_mem, t_final, solution_tmp, &t_now_tmp, CV_ONE_STEP );
                CVODECHKERR( comm, cvode_stat );
                // Interpolate the solution if the last step went over the prescribed final time
                if ( t_now_tmp > t_final ) {
                    cvode_stat = CVodeGetDky( cvode_mem, t_final, 0, solution_tmp );
                    CVODECHKERR( comm, cvode_stat );
                    t_now_tmp = t_final;
                }
                // Check that the temporary solution satisfies FSP tolerance
                sink_values = fsp->SinkStatesReduce( solution_tmp_dat );
                for ( int j = 0; j < sink_values.n_elem; ++j ) {
                    sink_values(j) = round2digit(sink_values(j));
                }
                double target_error = round2digit(fsp_tol * t_now_tmp / t_final);
                fsp_accept = ( PetscBool ) ( arma::max( sink_values )*double( fsp->get_num_constraints( )) <=  target_error);

                if ( !fsp_accept ) {
//                    arma::Row< PetscReal > sink_increase = sink_values - sink_values_old;
//                    arma::uvec i_sink_sorted = arma::sort_index( sink_increase, "descend" );
//                    PetscReal x1{0.0}, x2{round2digit(arma::sum(sink_increase) - (arma::sum( sink_values ) - target_error))};
//                    for ( auto i = 0; i < expand_sink.n_elem; ++i ) {
//                        if ( x1 > 1.1e0*x2 || sink_increase(i_sink_sorted(i)) == 0.0e0 ) {
//                            break;
//                        }
//                        expand_sink( i_sink_sorted( i )) = 1;
//                        x1 += sink_increase( i_sink_sorted( i ));
//                    }
                    arma::uvec iexpand = arma::find(sink_values > target_error/(double( fsp->get_num_constraints( ))));
                    expand_sink(iexpand).fill(1);

                    cvode_stat = CVodeGetDky( cvode_mem, t_now, 0, solution_tmp );
                    break;
                } else {
                    t_now = t_now_tmp;
                    if ( print_intermediate ) {
                        PetscPrintf( comm, "t_now = %.2e \n", t_now );
                    }
                    if ( logging ) {
                        perf_info.model_time[ perf_info.n_step ] = t_now;
                        petsc_err = VecGetSize( *solution, &perf_info.n_eqs[ size_t( perf_info.n_step ) ] );
                        CHKERRABORT( comm, petsc_err );
                        petsc_err = PetscTime( &perf_info.cpu_time[ perf_info.n_step ] );
                        CHKERRABORT( comm, petsc_err );
                        perf_info.n_step += 1;
                    }
                    sink_values_old = sink_values;
                }
            }
            // Copy data from temporary vector to solution vector
            CHKERRABORT( comm, VecCopy( solution_tmp_dat, *solution ));
            return expand_sink.max( );
        }

        int CVODEFSP::cvode_rhs( double t, N_Vector u, N_Vector udot, void *FPS ) {
            Vec udata = N_VGetVector_Petsc( u );
            Vec udotdata = N_VGetVector_Petsc( udot );
            PetscReal usum;
            VecNorm( udata, NORM_1, &usum );
            (( cme::parallel::OdeSolverBase * ) FPS )->RHSEval( t, udata, udotdata );
            VecNorm( udotdata, NORM_1, &usum );
            return 0;
        }

        int
        CVODEFSP::cvode_jac( N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr,
                             N_Vector tmp ) {
            Vec vdata = N_VGetVector_Petsc( v );
            Vec Jvdata = N_VGetVector_Petsc( Jv );
            (( cme::parallel::OdeSolverBase * ) FPS_ptr )->RHSEval( t, vdata, Jvdata );
            return 0;
        }

        void CVODEFSP::Free( ) {
            OdeSolverBase::Free( );
            if ( solution_tmp ) N_VDestroy( solution_tmp );
            if ( linear_solver ) SUNLinSolFree( linear_solver );
        }

        void CVODEFSP::SetCVodeTolerances( PetscReal _r_tol, PetscReal _abs_tol ) {
            rel_tol = _r_tol;
            abs_tol = _abs_tol;
        }

        CVODEFSP::~CVODEFSP( ) {
            if ( cvode_mem ) CVodeFree( &cvode_mem );
            Free( );
        }

    }
}