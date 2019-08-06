//
// Created by Huy Vo on 2019-07-14.
//

#include "TsFsp.h"

int pacmensl::TsFsp::TSRhsFunc(TS ts, PetscReal t, Vec u, Vec F, void *ctx)
{
  OdeSolverBase *ode_solver = ( OdeSolverBase * ) ctx;
  return ode_solver->EvaluateRHS(t, u, F);
}

pacmensl::TsFsp::TsFsp(MPI_Comm _comm) : OdeSolverBase(_comm)
{
}

PacmenslErrorCode pacmensl::TsFsp::SetUp()
{
  PacmenslErrorCode ierr;
  ierr = TSCreate(comm_, ts_.mem());
  PACMENSLCHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts_, NULL, &pacmensl::TsFsp::TSRhsFunc, ( void * ) this);
  PACMENSLCHKERRQ(ierr);
  ierr = TSSetType(ts_, type_);
  PACMENSLCHKERRQ(ierr);
  ierr = TSSetTolerances(ts_, abs_tol_, NULL, rel_tol_, NULL);
  PACMENSLCHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP);
  PACMENSLCHKERRQ(ierr);
  ierr = TSSetMaxTime(ts_, t_final_);
  PACMENSLCHKERRQ(ierr);
  PetscInt  event_direction[1] = {0};
  PetscBool event_terminate[1] = {PETSC_TRUE};
  ierr = TSSetEventHandler(ts_, 1, event_direction, event_terminate, &pacmensl::TsFsp::TSDetectFspError, NULL, this);
  PACMENSLCHKERRQ(ierr);
  ierr = TSSetFromOptions(ts_);
  PACMENSLCHKERRQ(ierr);
  return 0;
}

PetscInt pacmensl::TsFsp::Solve()
{
  PetscErrorCode petsc_err;
  petsc_err = VecDuplicate(*solution_, &solution_tmp_);
  CHKERRQ(petsc_err);
  petsc_err = VecCopy(*solution_, solution_tmp_);
  CHKERRQ(petsc_err);
  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  int stop = 0;
  petsc_err = TSSetTime(ts_, t_now_);
  CHKERRQ(petsc_err);
  petsc_err = TSSetSolution(ts_, solution_tmp_);
  CHKERRQ(petsc_err);
  petsc_err = TSSolve(ts_, solution_tmp_);
  CHKERRQ(petsc_err);
  petsc_err = TSGetSolveTime(ts_, &t_now_tmp);
  CHKERRQ(petsc_err);
//  while (t_now_ < t_final_)
//  {
//    petsc_err = TSStep(ts_); CHKERRQ(petsc_err);
//    petsc_err = TSGetSolveTime(ts_, &t_now_tmp); CHKERRQ(petsc_err);
//    // Interpolate the solution_ if the last step went over the prescribed final time
//    if (t_now_tmp > t_final_)
//    {
//      petsc_err = TSInterpolate(ts_, t_final_, solution_tmp_); CHKERRQ(petsc_err);
//      t_now_tmp = t_final_;
//    }
//    // Check that the temporary solution_ satisfies Fsp tolerance
//    if (stop_check_ != nullptr) stop = stop_check_(t_now_tmp, solution_tmp_, stop_data_);
//
//    if (stop == 1){
//      petsc_err = TSInterpolate(ts_, t_now_, solution_tmp_); CHKERRQ(petsc_err);
//      break;
//    } else{
//
//    }
//  }

  t_now_ = t_now_tmp;
  if (print_intermediate)
  {
    PetscPrintf(comm_, "t_now_ = %.2e \n", t_now_);
  }
  if (logging_enabled)
  {
    perf_info.model_time[perf_info.n_step] = t_now_;
    petsc_err = VecGetSize(*solution_, &perf_info.n_eqs[size_t(perf_info.n_step)]);
    CHKERRQ(petsc_err);
    petsc_err = PetscTime(&perf_info.cpu_time[perf_info.n_step]);
    CHKERRQ(petsc_err);
    perf_info.n_step += 1;
  }
  // Copy data from temporary vector to solution_ vector
  petsc_err = VecCopy(solution_tmp_, *solution_);
  CHKERRQ(petsc_err);
  stop = (t_now_ < t_final_);
  return stop;
}

PacmenslErrorCode pacmensl::TsFsp::FreeWorkspace()
{
  PacmenslErrorCode ierr;
  TSDestroy(ts_.mem());
  PACMENSLCHKERRQ(ierr);
  VecDestroy(&solution_tmp_);
  PACMENSLCHKERRQ(ierr);
  return OdeSolverBase::FreeWorkspace();
}

pacmensl::TsFsp::~TsFsp()
{
  FreeWorkspace();
}

int pacmensl::TsFsp::TSDetectFspError(TS ts, PetscReal t, Vec U, PetscScalar *fvalue, void *ctx)
{
  int  ierr;
  auto solver = ( TsFsp * ) ctx;
  ierr = solver->stop_check_(t, U, fvalue[0], solver->stop_data_);
  PACMENSLCHKERRQ(ierr);
//  VecView(U, PETSC_VIEWER_STDOUT_WORLD);
//  return -1;
  return 0;
}

