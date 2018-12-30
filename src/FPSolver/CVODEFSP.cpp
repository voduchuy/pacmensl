//
// Created by Huy Vo on 12/6/18.
//
#include <FPSolver/CVODEFSP.h>

#include "FiniteProblemSolver.h"
#include "CVODEFSP.h"

namespace cme{
    namespace parallel{

        CVODEFSP::CVODEFSP(MPI_Comm _comm, int lmm, int iter) : FiniteProblemSolver(_comm)
        {
            cvode_mem = CVodeCreate(lmm, iter);
            if (cvode_mem == nullptr){
                throw std::runtime_error("CVODE failed to initialize memory.\n");
            }
            solver_type = CVODE_BDF;
        }

        PetscInt CVODEFSP::Solve() {
            // Make sure the necessary data has been set
            assert(solution != nullptr);
            assert(rhs);
            assert(fsp != nullptr);

            PetscInt petsc_err;

            // N_Vector wrapper for the solution
            solution_wrapper = N_VMake_Petsc(solution);

            // Copy solution to the temporary solution
            solution_tmp = N_VClone(solution_wrapper);
            Vec* solution_tmp_dat = N_VGetVector_Petsc(solution_tmp);
            petsc_err = VecSetUp(*solution_tmp_dat);
            CHKERRABORT(comm, petsc_err);
            petsc_err = VecCopy(*solution, *solution_tmp_dat);
            CHKERRABORT(comm, petsc_err);

            // Set CVODE startiing time to the current timepoint
            t_now_tmp = t_now;

            // Initialize cvode
            cvode_stat = CVodeInit(cvode_mem, &cvode_rhs, t_now_tmp, solution_tmp); CVODECHKERR(comm, cvode_stat);
            cvode_stat = CVodeSetUserData(cvode_mem, (void*) this); CVODECHKERR(comm, cvode_stat);
            cvode_stat = CVodeSStolerances(cvode_mem, rel_tol, abs_tol); CVODECHKERR(comm, cvode_stat);
            cvode_stat = CVodeSetMaxNumSteps(cvode_mem, 10000); CVODECHKERR(comm, cvode_stat);
            cvode_stat = CVodeSetMaxConvFails(cvode_mem, 10000); CVODECHKERR(comm, cvode_stat);
            cvode_stat = CVodeSetMaxNonlinIters(cvode_mem, 10000); CVODECHKERR(comm, cvode_stat);

            // Create the linear solver without preconditioning
            linear_solver = SUNSPBCGS(solution_tmp, PREC_NONE, 0);
//            linear_solver = SUNSPGMR(solution_tmp, PREC_NONE, 30);
            cvode_stat = CVSpilsSetLinearSolver(cvode_mem, linear_solver); CVODECHKERR(comm, cvode_stat);
            cvode_stat = CVSpilsSetJacTimes(cvode_mem, NULL, &cvode_jac); CVODECHKERR(comm, cvode_stat);

            // Advance the temporary solution until either reaching final time or FSP error exceeding tolerance
            arma::Row<PetscReal> sink_values(fsp->GetNumSpecies());
            PetscBool fsp_accept;
            expand_sink.fill(0);
            while(t_now < t_final){
                cvode_stat = CVode(cvode_mem, t_final, solution_tmp, &t_now_tmp, CV_ONE_STEP); CVODECHKERR(comm, cvode_stat);
                // Interpolate the solution if the last step went over the prescribed final time
                if (t_now_tmp > t_final){
                    cvode_stat = CVodeGetDky(cvode_mem, t_final, 0, solution_tmp); CVODECHKERR(comm, cvode_stat);
                    t_now_tmp = t_final;
                }
                // Check that the temporary solution satisfies FSP tolerance
                while(1){
                    sink_values = fsp->SinkStatesReduce(*solution_tmp_dat);
                    fsp_accept = PETSC_TRUE;
                    for (auto i = 0; i < expand_sink.n_elem; ++i){
                        if (expand_sink.n_elem*sink_values(i) > fsp_tol*t_now_tmp/t_final){
                            fsp_accept = PETSC_FALSE;
                            expand_sink(i) = 1;
                        }
                    }
                    if (fsp_accept) break;
                    PetscReal tau = (t_now_tmp - t_now)/2.0;
                    cvode_stat = CVodeGetDky(cvode_mem, t_now + tau, 0, solution_tmp); CVODECHKERR(comm, cvode_stat);
                    t_now_tmp = t_now + tau;
                }
                // Copy data from temporary vector to solution vector
                t_now = t_now_tmp;
                CHKERRABORT(comm, VecCopy(*solution_tmp_dat, *solution));
                if (print_intermediate){
                    PetscPrintf(comm, "t_now = %.2e \n", t_now);
                }
                if (logging){
                    perf_info.model_time[perf_info.n_step] = t_now;
                    petsc_err = VecGetSize(*solution, &perf_info.n_eqs[size_t(perf_info.n_step)] );
                    CHKERRABORT(comm, petsc_err);
                    petsc_err = PetscTime(&perf_info.cpu_time[perf_info.n_step]);
                    CHKERRABORT(comm, petsc_err);
                    perf_info.n_step += 1;
                }
                if (expand_sink.max() != 0) break;
            }
            return expand_sink.max();
        }

        int CVODEFSP::cvode_rhs(double t, N_Vector u, N_Vector udot, void *FPS) {
            Vec* udata = N_VGetVector_Petsc(u);
            Vec* udotdata = N_VGetVector_Petsc(udot);
            PetscReal usum;
            VecNorm(*udata, NORM_1,&usum);
            ((cme::parallel::FiniteProblemSolver*) FPS)->RHSEval(t, *udata, *udotdata);
            VecNorm(*udotdata, NORM_1,&usum);
            return 0;
        }

        int
        CVODEFSP::cvode_jac(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr, N_Vector tmp) {
            Vec* vdata = N_VGetVector_Petsc(v);
            Vec* Jvdata = N_VGetVector_Petsc(Jv);
            ((cme::parallel::FiniteProblemSolver*) FPS_ptr)->RHSEval(t, *vdata, *Jvdata);
            return 0;
        }

        void CVODEFSP::Free() {
            FiniteProblemSolver::Free();
            if (solution_tmp) N_VDestroy(solution_tmp);
            if (linear_solver) SUNLinSolFree(linear_solver);
        }

        void CVODEFSP::SetCVodeTolerances(PetscReal _r_tol, PetscReal _abs_tol) {
            rel_tol = _r_tol;
            abs_tol = _abs_tol;
        }

        CVODEFSP::~CVODEFSP()   {
            if (cvode_mem) CVodeFree(&cvode_mem);
            Free();
        }

    }
}