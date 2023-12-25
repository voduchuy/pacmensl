/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <OdeSolver/CvodeFsp.h>
#include "OdeSolver/OdeSolverBase.h"

namespace pacmensl {

CvodeFsp::CvodeFsp(MPI_Comm _comm, int lmm) : OdeSolverBase(_comm) {
    lmm_ = lmm;
}

PetscInt CvodeFsp::Solve() {
    PacmenslErrorCode ierr;
    PetscErrorCode petsc_err;
    // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
    int stop = 0;
    PetscReal error_excess = 0.0;
    while (t_now_ < t_final_) {
        cvode_stat = CVode(cvode_mem, t_final_, cvode_solution, &t_now_tmp, CV_ONE_STEP);
        CVODECHKERRQ(cvode_stat);
        // Interpolate the solution_ if the last step went over the prescribed final time
        if (t_now_tmp > t_final_) {
            cvode_stat = CVodeGetDky(cvode_mem, t_final_, 0, cvode_solution);
            CVODECHKERRQ(cvode_stat);
            t_now_tmp = t_final_;
        }
        // Check that the temporary solution_ satisfies Fsp tolerance
        if (stop_check_ != nullptr) {
            ierr = stop_check_(t_now_tmp, cvode_solution_vec_wrapper, error_excess, stop_data_);
            PACMENSLCHKERRQ(ierr);
            if (error_excess > 0.0) {
                stop = 1;
                cvode_stat = CVodeGetDky(cvode_mem, t_now_, 0, cvode_solution);
                CVODECHKERRQ(cvode_stat);
                break;
            }
        }

        t_now_ = t_now_tmp;
        if (print_intermediate) {
            PetscPrintf(comm_, "t_now_ = %.2e \n", t_now_);
        }
        if (logging_enabled) {
            perf_info.model_time[perf_info.n_step] = t_now_;
            petsc_err = VecGetSize(*solution_, &perf_info.n_eqs[size_t(perf_info.n_step)]);
            CHKERRQ(petsc_err);
            petsc_err = PetscTime(&perf_info.cpu_time[perf_info.n_step]);
            CHKERRQ(petsc_err);
            perf_info.n_step += 1;
        }
    }
    // Copy data from temporary vector to solution_ vector
    petsc_err = VecCopy(cvode_solution_vec_wrapper, *solution_);
    CHKERRQ(petsc_err);
    return stop;
}

int CvodeFsp::eval_rhs_(double t, N_Vector u, N_Vector udot, void *solver) {
    int ierr{0};
    VecPlaceArray(workvec1, N_VGetArrayPointer(u));
    VecPlaceArray(workvec2, N_VGetArrayPointer(udot));
    ierr = EvaluateRHS(t, workvec1, workvec2);
    PACMENSLCHKERRQ(ierr);
    VecResetArray(workvec1);
    VecResetArray(workvec2);
    return ierr;
}

int
CvodeFsp::eval_jac_(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr,
                    N_Vector tmp) {
    int ierr{0};
    VecPlaceArray(workvec1, N_VGetArrayPointer(v));
    VecPlaceArray(workvec2, N_VGetArrayPointer(Jv));
    ierr = EvaluateRHS(t, workvec1, workvec2);
    PACMENSLCHKERRQ(ierr);
    VecResetArray(workvec1);
    VecResetArray(workvec2);
    return ierr;
}

int CvodeFsp::cvode_rhs(double t, N_Vector u, N_Vector udot, void *solver) {
    int ierr{0};
    auto solver_ptr = (pacmensl::CvodeFsp*) solver;
    return solver_ptr->eval_rhs_(t, u, udot, solver);
}

int
CvodeFsp::cvode_jac(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr,
                    N_Vector tmp) {
    auto solver_ptr = (pacmensl::CvodeFsp*) FPS_ptr;
    return solver_ptr->eval_jac_(v, Jv, t, u, fu, FPS_ptr, tmp);
}



int CvodeFsp::FreeWorkspace() {
    OdeSolverBase::FreeWorkspace();
    if (cvode_mem) CVodeFree(&cvode_mem);
    if (cvode_solution_vec_wrapper != nullptr) VecDestroy(&cvode_solution_vec_wrapper);
    if (workvec1 != nullptr) VecDestroy(&workvec1);
    if (workvec2 != nullptr) VecDestroy(&workvec2);
    if (cvode_solution != nullptr) N_VDestroy(cvode_solution);
    if (linear_solver != nullptr) SUNLinSolFree(linear_solver);
    cvode_solution = nullptr;
    linear_solver = nullptr;
    cvode_solution_vec_wrapper = nullptr;
    return 0;
}

CvodeFsp::~CvodeFsp() {
    FreeWorkspace();
}

PacmenslErrorCode CvodeFsp::SetUp() {
    // Make sure the necessary data has been set
    if (solution_ == nullptr) return -1;
    if (rhs_ == nullptr) return -1;

    PetscInt petsc_err;

    // Allocate memory for CVODE solution vector
    int local_length, global_length;
    petsc_err = VecGetLocalSize(*solution_, &local_length);
    CHKERRQ(petsc_err);
    petsc_err = VecGetSize(*solution_, &global_length);
    CHKERRQ(petsc_err);
    cvode_solution = N_VNew_Parallel(comm_, local_length, global_length);

    // Create PETSc Vec wrapper for CVODE solution vector
    PetscReal *cvode_vec_data = N_VGetArrayPointer(cvode_solution);
    petsc_err =
        VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, cvode_vec_data, &cvode_solution_vec_wrapper);
    CHKERRQ(petsc_err);

    // Create empty PETSc Vecs to handle right-hand-side evaluations
    petsc_err =
        VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, NULL, &workvec1);
    CHKERRQ(petsc_err);
    petsc_err =
        VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, NULL, &workvec2);
    CHKERRQ(petsc_err);

    // Copy initial condition to CVODE
    petsc_err = VecCopy(*solution_, cvode_solution_vec_wrapper);
    CHKERRQ(petsc_err);

    // Set CVODE starting time to the current timepoint
    t_now_tmp = t_now_;

    // Initialize cvode
    cvode_mem = CVodeCreate(lmm_);
    if (cvode_mem == nullptr) {
        PetscPrintf(comm_, "CVODE failed to initialize memory.\n");
        return -1;
    }

    cvode_stat = CVodeInit(cvode_mem, &cvode_rhs, t_now_tmp, cvode_solution);
    CVODECHKERRQ(cvode_stat);
    cvode_stat = CVodeSetUserData(cvode_mem, ( void * ) this);
    CVODECHKERRQ(cvode_stat);
    cvode_stat = CVodeSStolerances(cvode_mem, rel_tol_, abs_tol_);
    CVODECHKERRQ(cvode_stat);
    cvode_stat = CVodeSetMaxNumSteps(cvode_mem, 10000);
    CVODECHKERRQ(cvode_stat);
    cvode_stat = CVodeSetMaxConvFails(cvode_mem, 10000);
    CVODECHKERRQ(cvode_stat);
    cvode_stat = CVodeSetMaxNonlinIters(cvode_mem, 10000);
    CVODECHKERRQ(cvode_stat);

    // Create the linear solver without preconditioning
    linear_solver = SUNLinSol_SPGMR(cvode_solution, PREC_NONE, 100);
    cvode_stat = CVSpilsSetLinearSolver(cvode_mem, linear_solver);
    CVODECHKERRQ(cvode_stat);
    cvode_stat = CVSpilsSetJacTimes(cvode_mem, NULL, &cvode_jac);
    CVODECHKERRQ(cvode_stat);
    return 0;
}
}