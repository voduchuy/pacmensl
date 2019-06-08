//
// Created by Huy Vo on 12/6/18.
//
#include <OdeSolver/CvodeFsp.h>
#include "OdeSolver/OdeSolverBase.h"
#include "CvodeFsp.h"


namespace cme {
    namespace parallel {

        CvodeFsp::CvodeFsp( MPI_Comm _comm, int lmm ) : OdeSolverBase( _comm ) {
            cvode_mem = CVodeCreate( lmm);
            if ( cvode_mem == nullptr ) {
                throw std::runtime_error( "CVODE failed to initialize memory.\n" );
            }
            solver_type = CVODE_BDF;
        }

        PetscInt CvodeFsp::solve( ) {
            // Make sure the necessary data has been set
            assert( solution_ != nullptr );
            assert( rhs_ );


            PetscInt petsc_err;
            // N_Vector wrapper for the solution_
            solution_wrapper = N_VMake_Petsc( *solution_ );

            // Copy solution_ to the temporary solution_
            solution_tmp = N_VClone( solution_wrapper );
            Vec solution_tmp_dat = N_VGetVector_Petsc( solution_tmp );
            petsc_err = VecSetUp( solution_tmp_dat );
            CHKERRABORT( comm_, petsc_err );
            petsc_err = VecCopy( *solution_, solution_tmp_dat );
            CHKERRABORT( comm_, petsc_err );

            // Set CVODE starting time to the current timepoint
            t_now_tmp = t_now_;

            // Initialize cvode
            cvode_stat = CVodeInit( cvode_mem, &cvode_rhs, t_now_tmp, solution_tmp );
            CVODECHKERR( comm_, cvode_stat );
            cvode_stat = CVodeSetUserData( cvode_mem, ( void * ) this );
            CVODECHKERR( comm_, cvode_stat );
            cvode_stat = CVodeSStolerances( cvode_mem, rel_tol, abs_tol );
            CVODECHKERR( comm_, cvode_stat );
            cvode_stat = CVodeSetMaxNumSteps( cvode_mem, 10000 );
            CVODECHKERR( comm_, cvode_stat );
            cvode_stat = CVodeSetMaxConvFails( cvode_mem, 10000 );
            CVODECHKERR( comm_, cvode_stat );
            cvode_stat = CVodeSetMaxNonlinIters( cvode_mem, 10000 );
            CVODECHKERR( comm_, cvode_stat );

            // Create the linear solver without preconditioning
            linear_solver = SUNSPBCGS( solution_tmp, PREC_NONE, 0 );
            cvode_stat = CVSpilsSetLinearSolver( cvode_mem, linear_solver );
            CVODECHKERR( comm_, cvode_stat );
            cvode_stat = CVSpilsSetJacTimes( cvode_mem, NULL, &cvode_jac );
            CVODECHKERR( comm_, cvode_stat );

            // Advance the temporary solution_ until either reaching final time or FSP error exceeding tolerance
            int stop = 0;
            while ( t_now_ < t_final_ ) {
                cvode_stat = CVode( cvode_mem, t_final_, solution_tmp, &t_now_tmp, CV_ONE_STEP );
                CVODECHKERR( comm_, cvode_stat );
                // Interpolate the solution_ if the last step went over the prescribed final time
                if ( t_now_tmp > t_final_ ) {
                    cvode_stat = CVodeGetDky( cvode_mem, t_final_, 0, solution_tmp );
                    CVODECHKERR( comm_, cvode_stat );
                    t_now_tmp = t_final_;
                }
                // Check that the temporary solution_ satisfies FSP tolerance
                if (stop_check_ != nullptr) stop = stop_check_(t_now_tmp, solution_tmp_dat, stop_data_);
                if ( stop == 1) {
                    cvode_stat = CVodeGetDky( cvode_mem, t_now_, 0, solution_tmp );
                    break;
                } else {
                    t_now_ = t_now_tmp;
                    if ( print_intermediate ) {
                        PetscPrintf( comm_, "t_now_ = %.2e \n", t_now_ );
                    }
                    if ( logging ) {
                        perf_info.model_time[ perf_info.n_step ] = t_now_;
                        petsc_err = VecGetSize( *solution_, &perf_info.n_eqs[ size_t( perf_info.n_step ) ] );
                        CHKERRABORT( comm_, petsc_err );
                        petsc_err = PetscTime( &perf_info.cpu_time[ perf_info.n_step ] );
                        CHKERRABORT( comm_, petsc_err );
                        perf_info.n_step += 1;
                    }
                }
            }
            // Copy data from temporary vector to solution_ vector
            CHKERRABORT( comm_, VecCopy( solution_tmp_dat, *solution_ ));
            return stop;
        }

        int CvodeFsp::cvode_rhs( double t, N_Vector u, N_Vector udot, void *solver ) {
            Vec udata = N_VGetVector_Petsc( u );
            Vec udotdata = N_VGetVector_Petsc( udot );
            PetscReal usum;
            VecNorm( udata, NORM_1, &usum );
            (( cme::parallel::OdeSolverBase * ) solver )->evaluate_rhs( t, udata, udotdata );
            VecNorm( udotdata, NORM_1, &usum );
            return 0;
        }

        int
        CvodeFsp::cvode_jac( N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr,
                             N_Vector tmp ) {
            Vec vdata = N_VGetVector_Petsc( v );
            Vec Jvdata = N_VGetVector_Petsc( Jv );
            (( cme::parallel::OdeSolverBase * ) FPS_ptr )->evaluate_rhs( t, vdata, Jvdata );
            return 0;
        }

        void CvodeFsp::free( ) {
            OdeSolverBase::free( );
            if ( solution_tmp ) N_VDestroy( solution_tmp );
            if ( linear_solver ) SUNLinSolFree( linear_solver );
        }

        void CvodeFsp::SetCVodeTolerances( PetscReal _r_tol, PetscReal _abs_tol ) {
            rel_tol = _r_tol;
            abs_tol = _abs_tol;
        }

        CvodeFsp::~CvodeFsp( ) {
            if ( cvode_mem ) CVodeFree( &cvode_mem );
            free( );
        }
    }
}