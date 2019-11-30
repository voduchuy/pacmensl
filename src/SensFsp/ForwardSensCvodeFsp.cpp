//
// Created by Huy Vo on 2019-06-28.
//

#include "ForwardSensCvodeFsp.h"

int pacmensl::ForwardSensCvodeFsp::cvode_rhs(double t, N_Vector u, N_Vector udot, void *data) {
  int ierr{0};
  Vec udata    = N_VGetVector_Petsc(u);
  Vec udotdata = N_VGetVector_Petsc(udot);
  ierr = (( pacmensl::ForwardSensSolverBase * ) data)->EvaluateRHS(t, udata, udotdata);
  PACMENSLCHKERRQ(ierr);
  return 0;
}

int pacmensl::ForwardSensCvodeFsp::cvode_jac(N_Vector v,
                                             N_Vector Jv,
                                             realtype t,
                                             N_Vector u,
                                             N_Vector fu,
                                             void *FPS_ptr,
                                             N_Vector tmp) {
  int ierr{0};
  Vec vdata  = N_VGetVector_Petsc(v);
  Vec Jvdata = N_VGetVector_Petsc(Jv);
  ierr = (( pacmensl::ForwardSensSolverBase * ) FPS_ptr)->EvaluateRHS(t, vdata, Jvdata);
  PACMENSLCHKERRQ(ierr);
  return ierr;
}

int pacmensl::ForwardSensCvodeFsp::cvsens_rhs(int Ns,
                                              PetscReal t,
                                              N_Vector y,
                                              N_Vector ydot,
                                              int iS,
                                              N_Vector yS,
                                              N_Vector ySdot,
                                              void *user_data,
                                              N_Vector tmp1,
                                              N_Vector tmp2) {
  if (iS >= Ns) return -1;
  int  ierr{0};
  auto my_solver = ( pacmensl::ForwardSensSolverBase * ) user_data;
  ierr = my_solver->EvaluateRHS(t, N_VGetVector_Petsc(yS), N_VGetVector_Petsc(tmp1));
  PACMENSLCHKERRQ(ierr);
  ierr = my_solver->EvaluateSensRHS(iS, t, N_VGetVector_Petsc(y), N_VGetVector_Petsc(tmp2));
  PACMENSLCHKERRQ(ierr);

  ierr = VecSet(N_VGetVector_Petsc(ySdot), 0.0);
  PACMENSLCHKERRQ(ierr);
  ierr = VecAXPY(N_VGetVector_Petsc(ySdot), 1.0, N_VGetVector_Petsc(tmp1));
  PACMENSLCHKERRQ(ierr);
  ierr = VecAXPY(N_VGetVector_Petsc(ySdot), 1.0, N_VGetVector_Petsc(tmp2));
  PACMENSLCHKERRQ(ierr);
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensCvodeFsp::SetUp() {
  PacmenslErrorCode ierr;
  // Make sure the necessary data has been set
  if (solution_ == nullptr) return -1;
  if (rhs_ == nullptr) return -1;
  if (srhs_ == nullptr) return -1;

  PetscInt petsc_err;

  // N_Vector wrapper for the solution_
  solution_wrapper = N_VMake_Petsc(*solution_);
  sens_vecs_wrapper.resize(num_parameters_);
  for (int i{0}; i < num_parameters_; ++i) {
    sens_vecs_wrapper[i] = N_VMake_Petsc(*sens_vecs_[i]);
  }

  // Copy solution_ to the temporary solution_
  solution_tmp = N_VClone_Petsc(solution_wrapper);
  petsc_err    = VecCopy(*solution_, N_VGetVector_Petsc(solution_tmp));
  CHKERRQ(petsc_err);
  sens_vecs_tmp = N_VCloneVectorArray_Petsc(num_parameters_, solution_tmp);
  for (int i{0}; i < num_parameters_; ++i) {
    petsc_err = VecCopy(*sens_vecs_[i], N_VGetVector_Petsc(sens_vecs_tmp[i]));
    CHKERRQ(petsc_err);
  }

  // Set CVODE starting time to the current timepoint
  t_now_tmp_ = t_now_;

  // Initialize cvode
  cvode_mem = CVodeCreate(lmm_);
  if (cvode_mem == nullptr) {
    PetscPrintf(comm_, "CVODE failed to initialize memory.\n");
    return -1;
  }

  cvode_stat = CVodeInit(cvode_mem, &cvode_rhs, t_now_tmp_, solution_tmp);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSetUserData(cvode_mem, ( void * ) this);
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
//  linear_solver = SUNSPBCGS(solution_tmp, PREC_NONE, 0);
  linear_solver = SUNSPGMR(solution_tmp, PREC_NONE, 50);
  if (linear_solver == nullptr) {
    PetscPrintf(comm_, "CVODE failed to initialize memory.\n");
    return -1;
  }
  cvode_stat = CVSpilsSetLinearSolver(cvode_mem, linear_solver);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVSpilsSetJacTimes(cvode_mem, nullptr, &cvode_jac);
  CVODECHKERRQ(cvode_stat);

  // Define the sensitivity problem
  cvode_stat = CVodeSensInit1(cvode_mem, num_parameters_, CV_STAGGERED1, cvsens_rhs, &sens_vecs_tmp[0]);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVodeSetSensErrCon(cvode_mem, SUNTRUE);
  CVODECHKERRQ(cvode_stat);

  cvode_stat = CVodeSensEEtolerances(cvode_mem);
  CVODECHKERRQ(cvode_stat);
//  std::vector<double> abs_tols(sens_vecs_.size(), 1.0e-4);
//  epic_stat = CVodeSensSStolerances(cvode_mem, rel_tol, &abs_tols[0]);
//  CVODECHKERRQ(epic_stat);
  set_up_ = true;

  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensCvodeFsp::FreeWorkspace() {
  int ierr;
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (linear_solver != nullptr) SUNLinSolFree(linear_solver);
  if (solution_wrapper != nullptr) N_VDestroy(solution_wrapper);
  for (int i{0}; i < sens_vecs_wrapper.size(); ++i) {
    N_VDestroy(sens_vecs_wrapper[i]);
  }
  sens_vecs_wrapper.clear();
  if (solution_tmp != nullptr) N_VDestroy(solution_tmp);
  N_VDestroyVectorArray_Petsc(sens_vecs_tmp, num_parameters_);

  solution_wrapper = nullptr;
  sens_vecs_tmp    = nullptr;
  solution_tmp     = nullptr;
  linear_solver    = nullptr;
  num_parameters_  = 0;

  set_up_ = false;
  return ForwardSensSolverBase::FreeWorkspace();
}

PetscInt pacmensl::ForwardSensCvodeFsp::Solve() {
  if (!set_up_) SetUp();
  PetscErrorCode   petsc_err;
  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  int              stop = 0;
  std::vector<Vec> sens_vecs_tmp_dat(num_parameters_);
  for (int         i{0}; i < sens_vecs_tmp_dat.size(); ++i) {
    sens_vecs_tmp_dat[i] = N_VGetVector_Petsc(sens_vecs_tmp[i]);
  }
  while (t_now_ < t_final_) {
    cvode_stat = CVode(cvode_mem, t_final_, solution_tmp, &t_now_tmp_, CV_ONE_STEP);
    CVODECHKERRQ(cvode_stat);
    // Interpolate the solution_ if the last step went over the prescribed final time
    if (t_now_tmp_ > t_final_) {
      cvode_stat = CVodeGetDky(cvode_mem, t_final_, 0, solution_tmp);
      CVODECHKERRQ(cvode_stat);
      t_now_tmp_ = t_final_;
    }
    cvode_stat = CVodeGetSensDky(cvode_mem, t_now_tmp_, 0, sens_vecs_tmp);
    CVODECHKERRQ(cvode_stat);

    // Check that the temporary solution_ satisfies Fsp tolerance
    if (stop_check_ != nullptr) {
      stop = stop_check_(t_now_tmp_,
                         N_VGetVector_Petsc(solution_tmp),
                         num_parameters_,
                         sens_vecs_tmp_dat.data(),
                         stop_data_);
    }
    if (stop == 1) {
      cvode_stat = CVodeGetDky(cvode_mem, t_now_, 0, solution_tmp);
      CVODECHKERRQ(cvode_stat);
      cvode_stat = CVodeGetSensDky(cvode_mem, t_now_, 0, sens_vecs_tmp);
      CVODECHKERRQ(cvode_stat);
      break;
    } else {
      t_now_ = t_now_tmp_;
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
  petsc_err = VecCopy(N_VGetVector_Petsc(solution_tmp), *solution_);
  CHKERRQ(petsc_err);
  for (int i{0}; i < num_parameters_; ++i) {
    petsc_err = VecCopy(N_VGetVector_Petsc(sens_vecs_tmp[i]), *sens_vecs_[i]);
    CHKERRQ(petsc_err);
  }
  return stop;
}

pacmensl::ForwardSensCvodeFsp::~ForwardSensCvodeFsp() {
  FreeWorkspace();
}
