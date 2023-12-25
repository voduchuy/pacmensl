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

#include "ForwardSensCvodeFsp.h"

int pacmensl::ForwardSensCvodeFsp::cvode_rhs(double t, N_Vector u, N_Vector udot, void *data) {
  int ierr{0};
  auto solver_ptr = (pacmensl::ForwardSensCvodeFsp*) data;
  return solver_ptr->eval_rhs_(t, u, udot, data);
}

int pacmensl::ForwardSensCvodeFsp::eval_rhs_(double t, N_Vector u, N_Vector udot, void *solver) {
  int ierr{0};
  VecPlaceArray(workvec1, N_VGetArrayPointer(u));
  VecPlaceArray(workvec2, N_VGetArrayPointer(udot));
  ierr = EvaluateRHS(t, workvec1, workvec2);
  PACMENSLCHKERRQ(ierr);
  VecResetArray(workvec1);
  VecResetArray(workvec2);
  return ierr;
}

int pacmensl::ForwardSensCvodeFsp::cvode_jac(N_Vector v,
                                             N_Vector Jv,
                                             realtype t,
                                             N_Vector u,
                                             N_Vector fu,
                                             void *FPS_ptr,
                                             N_Vector tmp) {
  auto solver_ptr = (pacmensl::ForwardSensCvodeFsp*) FPS_ptr;
  return solver_ptr->eval_jac_(v, Jv, t, u, fu, FPS_ptr, tmp);
}

int
pacmensl::ForwardSensCvodeFsp::eval_jac_(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu, void *FPS_ptr,
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
  auto my_solver = ( pacmensl::ForwardSensCvodeFsp * ) user_data;
  return my_solver->eval_sens_rhs_(Ns, t, y, ydot, iS, yS, ySdot, user_data, tmp1, tmp2);
}

int pacmensl::ForwardSensCvodeFsp::eval_sens_rhs_(int Ns,
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

  VecPlaceArray(workvec1, N_VGetArrayPointer(yS));
  VecPlaceArray(workvec2, N_VGetArrayPointer(y));

  // Set workvec3 = tmp1 and workvec4 = tmp2
  VecPlaceArray(workvec3, N_VGetArrayPointer(tmp1));
  VecPlaceArray(workvec4, N_VGetArrayPointer(tmp2));

  ierr = EvaluateRHS(t, workvec1, workvec3);
  PACMENSLCHKERRQ(ierr);
  ierr = EvaluateSensRHS(iS, t, workvec2, workvec4);
  PACMENSLCHKERRQ(ierr);

  // Set workvec1 = ySdot
  VecResetArray(workvec1);
  VecPlaceArray(workvec1, N_VGetArrayPointer(ySdot));
  ierr = VecSet(workvec1, 0.0);
  PACMENSLCHKERRQ(ierr);
  ierr = VecAXPY(workvec1, 1.0, workvec3);
  PACMENSLCHKERRQ(ierr);
  ierr = VecAXPY(workvec1, 1.0, workvec4);
  PACMENSLCHKERRQ(ierr);

  VecResetArray(workvec1);
  VecResetArray(workvec2);
  VecResetArray(workvec3);
  VecResetArray(workvec4);
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensCvodeFsp::SetUp() {
  PacmenslErrorCode ierr;
  // Make sure the necessary data has been set
  if (solution_ == nullptr) return -1;
  if (rhs_ == nullptr) return -1;
  if (srhs_ == nullptr) return -1;

  PetscInt petsc_err;

  // Allocate memory for CVODE solution vector
  int local_length, global_length;
  petsc_err = VecGetLocalSize(*solution_, &local_length);
  CHKERRQ(petsc_err);
  petsc_err = VecGetSize(*solution_, &global_length);
  CHKERRQ(petsc_err);
  cvodes_solution_vec = N_VNew_Parallel(comm_, local_length, global_length);
  cvodes_sens_vecs.resize(num_parameters_);
  for (int i{0}; i < num_parameters_; ++i) {
    cvodes_sens_vecs[i] = N_VNew_Parallel(comm_, local_length, global_length);
  }

  // Create PETSc Vec wrapper for CVODE solution vector
  PetscReal *cvode_vec_data = N_VGetArrayPointer(cvodes_solution_vec);
  petsc_err =
      VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, cvode_vec_data, &cvodes_solution_vec_wrapper);
  CHKERRQ(petsc_err);
  cvodes_sens_vec_wrappers.resize(num_parameters_);
  for (int i{0}; i < num_parameters_; ++i){
    cvode_vec_data = N_VGetArrayPointer(cvodes_sens_vecs[i]);
    petsc_err =
        VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, cvode_vec_data, &cvodes_sens_vec_wrappers[i]);
    CHKERRQ(petsc_err);
  }

  // Create empty PETSc Vecs to handle right-hand-side evaluations
  petsc_err =
      VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, NULL, &workvec1);
  CHKERRQ(petsc_err);
  petsc_err =
      VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, NULL, &workvec2);
  CHKERRQ(petsc_err);
  petsc_err =
      VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, NULL, &workvec3);
  CHKERRQ(petsc_err);
  petsc_err =
      VecCreateMPIWithArray(comm_, 1, local_length, PETSC_DECIDE, NULL, &workvec4);
  CHKERRQ(petsc_err);

  // Copy solution_ to the temporary solution_
  petsc_err = VecCopy(*solution_, cvodes_solution_vec_wrapper);
  CHKERRQ(petsc_err);
  for (int i{0}; i < num_parameters_; ++i) {
    petsc_err = VecCopy(*sens_vecs_[i], cvodes_sens_vec_wrappers[i]);
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

  cvode_stat = CVodeInit(cvode_mem, &cvode_rhs, t_now_tmp_, cvodes_solution_vec);
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
//  linear_solver = SUNSPBCGS(cvode_solution, PREC_NONE, 0);
  linear_solver = SUNSPGMR(cvodes_solution_vec, PREC_NONE, 50);
  if (linear_solver == nullptr) {
    PetscPrintf(comm_, "CVODE failed to initialize memory.\n");
    return -1;
  }
  cvode_stat = CVSpilsSetLinearSolver(cvode_mem, linear_solver);
  CVODECHKERRQ(cvode_stat);
  cvode_stat = CVSpilsSetJacTimes(cvode_mem, nullptr, &cvode_jac);
  CVODECHKERRQ(cvode_stat);

  // Define the sensitivity problem
  cvode_stat = CVodeSensInit1(cvode_mem, num_parameters_, CV_STAGGERED1, cvsens_rhs, &cvodes_sens_vecs[0]);
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

PetscInt pacmensl::ForwardSensCvodeFsp::Solve() {
  if (!set_up_) SetUp();
  PetscErrorCode   petsc_err;
  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  int              stop = 0;
  while (t_now_ < t_final_) {
    cvode_stat = CVode(cvode_mem, t_final_, cvodes_solution_vec, &t_now_tmp_, CV_ONE_STEP);
    CVODECHKERRQ(cvode_stat);
    // Interpolate the solution_ if the last step went over the prescribed final time
    if (t_now_tmp_ > t_final_) {
      cvode_stat = CVodeGetDky(cvode_mem, t_final_, 0, cvodes_solution_vec);
      CVODECHKERRQ(cvode_stat);
      t_now_tmp_ = t_final_;
    }
    cvode_stat = CVodeGetSensDky(cvode_mem, t_now_tmp_, 0,cvodes_sens_vecs.data());
    CVODECHKERRQ(cvode_stat);

    // Check that the temporary solution_ satisfies Fsp tolerance
    if (stop_check_ != nullptr) {
      stop = stop_check_(t_now_tmp_,
                         cvodes_solution_vec_wrapper,
                         num_parameters_,
                         cvodes_sens_vec_wrappers.data(),
                         stop_data_);
    }
    if (stop == 1) {
      cvode_stat = CVodeGetDky(cvode_mem, t_now_, 0, cvodes_solution_vec);
      CVODECHKERRQ(cvode_stat);
      cvode_stat = CVodeGetSensDky(cvode_mem, t_now_, 0, cvodes_sens_vecs.data());
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
  petsc_err = VecCopy(cvodes_solution_vec_wrapper, *solution_);
  CHKERRQ(petsc_err);
  for (int i{0}; i < num_parameters_; ++i) {
    petsc_err = VecCopy(cvodes_sens_vec_wrappers[i], *sens_vecs_[i]);
    CHKERRQ(petsc_err);
  }
  return stop;
}

PacmenslErrorCode pacmensl::ForwardSensCvodeFsp::FreeWorkspace() {
  int ierr;
  if (cvode_mem) CVodeFree(&cvode_mem);
  if (linear_solver != nullptr) SUNLinSolFree(linear_solver);
  if (cvodes_solution_vec_wrapper != nullptr) VecDestroy(&cvodes_solution_vec_wrapper);
  for (int i{0}; i < cvodes_sens_vec_wrappers.size(); ++i){
    VecDestroy(&cvodes_sens_vec_wrappers[i]);
  }
  cvodes_sens_vec_wrappers.clear();
  if (cvodes_solution_vec != nullptr) N_VDestroy(cvodes_solution_vec);
  for (int i{0}; i < cvodes_sens_vecs.size(); ++i) {
    N_VDestroy(cvodes_sens_vecs[i]);
  }
  cvodes_sens_vecs.clear();
  if (workvec1 != nullptr) VecDestroy(&workvec1);
  if (workvec2 != nullptr) VecDestroy(&workvec2);
  if (workvec3 != nullptr) VecDestroy(&workvec3);
  if (workvec4 != nullptr) VecDestroy(&workvec4);

  cvodes_solution_vec = nullptr;
  linear_solver    = nullptr;
  num_parameters_  = 0;

  set_up_ = false;
  return ForwardSensSolverBase::FreeWorkspace();
}

pacmensl::ForwardSensCvodeFsp::~ForwardSensCvodeFsp() {
  FreeWorkspace();
}
