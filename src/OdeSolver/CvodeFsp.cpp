//
// Created by Huy Vo on 12/6/18.
//
#include <OdeSolver/CvodeFsp.h>
#include "OdeSolver/OdeSolverBase.h"
#include "CvodeFsp.h"

namespace pacmensl {

CvodeFsp::CvodeFsp(MPI_Comm _comm, int lmm) : OdeSolverBase(_comm) {
  lmm_ = lmm;
}

PetscInt CvodeFsp::Solve() {
  // Make sure the necessary data has been set
  if (solution_ == nullptr) return -1;
  if (rhs_ == nullptr) return -1;

  PetscInt petsc_err;
  // N_Vector wrapper for the solution_
  solution_wrapper = N_VMake_Petsc(*solution_);

  // Copy solution_ to the temporary solution_
  solution_tmp = N_VClone(solution_wrapper);
  Vec solution_tmp_dat = N_VGetVector_Petsc(solution_tmp);
  petsc_err = VecCopy(*solution_, solution_tmp_dat);
  CHKERRQ(petsc_err);

  // Set CVODE starting time to the current timepoint
  t_now_tmp = t_now_;

  // Initialize cvode
  cvode_mem = CVodeCreate(lmm_);
  if (cvode_mem == nullptr) {
    PetscPrintf(comm_, "CVODE failed to initialize memory.\n");
    return -1;
  }
  cvode_stat = CVodeInit(cvode_mem, &cvode_rhs, t_now_tmp, solution_tmp);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSetUserData(cvode_mem, (void *) this);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSStolerances(cvode_mem, rel_tol, abs_tol);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSetMaxNumSteps(cvode_mem, 10000);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSetMaxConvFails(cvode_mem, 10000);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSetMaxNonlinIters(cvode_mem, 10000);
  CVODECHKERRQ(cvode_stat);

  // Create the linear solver without preconditioning
  linear_solver = SUNSPBCGS(solution_tmp, PREC_NONE, 0);
  cvode_stat = CVSpilsSetLinearSolver(cvode_mem, linear_solver);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVSpilsSetJacTimes(cvode_mem, NULL, &cvode_jac);
  CVODECHKERRQ(cvode_stat);

  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  int stop = 0;
  while (t_now_ < t_final_) {
    cvode_stat = CVode(cvode_mem, t_final_, solution_tmp, &t_now_tmp, CV_ONE_STEP);
    CVODECHKERRQ(cvode_stat);
    // Interpolate the solution_ if the last step went over the prescribed final time
    if (t_now_tmp > t_final_) {
      cvode_stat = CVodeGetDky(cvode_mem, t_final_, 0, solution_tmp);
      CVODECHKERRQ(cvode_stat);
      t_now_tmp = t_final_;
    }
    // Check that the temporary solution_ satisfies Fsp tolerance
    if (stop_check_ != nullptr) stop = stop_check_(t_now_tmp, solution_tmp_dat, stop_data_);
    if (stop == 1) {
      cvode_stat = CVodeGetDky(cvode_mem, t_now_, 0, solution_tmp);
      break;
    } else {
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
  }
  // Copy data from temporary vector to solution_ vector
  petsc_err = VecCopy(solution_tmp_dat, *solution_);
  CHKERRQ(petsc_err);
  int ierr = FreeWorkspace();
  PACMENSLCHKERRQ(ierr);
  return stop;
}

int CvodeFsp::cvode_rhs(double t, N_Vector u, N_Vector udot, void *solver) {
  int ierr{0};
  Vec udata = N_VGetVector_Petsc(u);
  Vec udotdata = N_VGetVector_Petsc(udot);
  ierr = ((pacmensl::OdeSolverBase *) solver)->EvaluateRHS(t, udata, udotdata);
  PACMENSLCHKERRQ(ierr);
  return ierr;
}

int
CvodeFsp::cvode_jac(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr,
                    N_Vector tmp) {
  int ierr{0};
  Vec vdata = N_VGetVector_Petsc(v);
  Vec Jvdata = N_VGetVector_Petsc(Jv);
  ierr =  ((pacmensl::OdeSolverBase *) FPS_ptr)->EvaluateRHS(t, vdata, Jvdata);
  PACMENSLCHKERRQ(ierr);
  return ierr;
}

int CvodeFsp::FreeWorkspace() {
  OdeSolverBase::FreeWorkspace();
  int ierr;
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (solution_tmp != nullptr) N_VDestroy(solution_tmp);
  if (linear_solver != nullptr ) SUNLinSolFree(linear_solver);
  solution_tmp = nullptr;
  linear_solver = nullptr;
  return 0;
}

int CvodeFsp::SetCVodeTolerances(PetscReal _r_tol, PetscReal _abs_tol) {
  rel_tol = _r_tol;
  abs_tol = _abs_tol;
  return 0;
}

CvodeFsp::~CvodeFsp() {
  FreeWorkspace();
}
}