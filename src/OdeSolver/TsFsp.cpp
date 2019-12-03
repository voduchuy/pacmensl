//
// Created by Huy Vo on 2019-07-14.
//

#include "TsFsp.h"

pacmensl::TsFsp::TsFsp(MPI_Comm _comm) : OdeSolverBase(_comm)
{
}

PacmenslErrorCode pacmensl::TsFsp::SetUp()
{
  PacmenslErrorCode ierr;
  int implicit;
  TSType ts_type;

  ierr = TSCreate(comm_,ts_.mem());
  CHKERRQ(ierr);
  ierr = TSSetType(ts_,type_.c_str());
  CHKERRQ(ierr);

  ierr = TSSetTolerances(ts_,abs_tol_,NULL,rel_tol_,NULL);
  CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts_,TS_EXACTFINALTIME_MATCHSTEP);
  CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts_,( void * ) this);
  CHKERRQ(ierr);
  ierr = TSSetPostEvaluate(ts_,&TSCheckFspError);
  CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts_,100000);
  CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts_);
  CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts_, -1);
  CHKERRQ(ierr);
  ierr = TSGetType(ts_, &ts_type);
  CHKERRQ(ierr);

  ierr = CheckImplicitType(ts_type, &implicit);
  CHKERRQ(ierr);

  if (implicit == 1){
    ierr = fspmat_->CreateRHSJacobian(&J);
    CHKERRQ(ierr);
    ierr = TSSetIFunction(ts_,NULL,&pacmensl::TsFsp::TSIFunc,( void * ) this);
    CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts_,J,J,pacmensl::TsFsp::TSIJacFunc,( void * ) this);
    CHKERRQ(ierr);
  }
  else{
    ierr = TSSetRHSFunction(ts_,NULL,&pacmensl::TsFsp::TSRhsFunc,( void * ) this);
    CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts_,J,J,TSJacFunc,( void * ) this);
    CHKERRQ(ierr);
  }

  return 0;
}

PetscInt pacmensl::TsFsp::Solve()
{
  PetscErrorCode petsc_err;

  petsc_err = VecDuplicate(*solution_,&solution_tmp_);
  CHKERRQ(petsc_err);
  petsc_err = VecCopy(*solution_,solution_tmp_);
  CHKERRQ(petsc_err);
  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  fsp_stop_ = 0;

  petsc_err = TSSetTime(ts_,t_now_);
  CHKERRQ(petsc_err);
  petsc_err = TSSetMaxTime(ts_,t_final_);
  CHKERRQ(petsc_err);
  petsc_err = TSSetSolution(ts_,solution_tmp_);
  CHKERRQ(petsc_err);
  petsc_err = TSSolve(ts_,PETSC_NULL);
  CHKERRQ(petsc_err);

  // Copy data from temporary vector to solution_ vector
  petsc_err = VecCopy(solution_tmp_,*solution_);
  CHKERRQ(petsc_err);
  return fsp_stop_;
}

PacmenslErrorCode pacmensl::TsFsp::FreeWorkspace()
{
  PacmenslErrorCode ierr = 0;
  if (J != nullptr)
  {
    MatDestroy(&J);
    J = nullptr;
  }
  njac                   = 0;
  nstep                  = 0;
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

int pacmensl::TsFsp::TSCheckFspError(TS ts)
{
  void      *ts_ctx;
  PetscInt  ierr;
  PetscReal err{0.0},t_now_tmp;
  Vec       solution_tmp_;

  TSGetApplicationContext(ts,&ts_ctx);
  auto tsfsp_ctx = ( TsFsp * ) ts_ctx;

  ierr = TSGetSolution(ts,&solution_tmp_);
  CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t_now_tmp);
  CHKERRQ(ierr);
  tsfsp_ctx->nstep += 1;

  // Check that the temporary solution_ satisfies Fsp tolerance
  if (tsfsp_ctx->stop_check_ != nullptr)
  {
    ierr = tsfsp_ctx->stop_check_(t_now_tmp,solution_tmp_,err,tsfsp_ctx->stop_data_);
    PACMENSLCHKERRQ(ierr);
    if (err > 0.0)
    {
      tsfsp_ctx->fsp_stop_ = 1;
      PetscReal err2{1.0};
      PetscInt ntrial{0};
      while (ntrial < 10 && err2 > 0.0){
        // Try halving the stepsize
        t_now_tmp = tsfsp_ctx->t_now_ + 0.5*(t_now_tmp - tsfsp_ctx->t_now_);
        ierr = TSInterpolate(ts, t_now_tmp,solution_tmp_);
        CHKERRQ(ierr);
        ierr = tsfsp_ctx->stop_check_(t_now_tmp,solution_tmp_,err2,tsfsp_ctx->stop_data_);
        PACMENSLCHKERRQ(ierr);
        ntrial+=1;
      }
      if (ntrial >= 10){
        ierr = TSInterpolate(ts, tsfsp_ctx->t_now_,solution_tmp_);
        CHKERRQ(ierr);
      }
      else{
        tsfsp_ctx->t_now_ = t_now_tmp;
      }
      ierr = TSSetConvergedReason(ts,TS_CONVERGED_USER);
      CHKERRQ(ierr);
      return 0;
    }
  }

  if (tsfsp_ctx->print_intermediate)
  {
    PetscPrintf(tsfsp_ctx->comm_,
                "t_now_ = %.2e stepsize = %.2e nstep = %d njac = %d \n",
                t_now_tmp,
                t_now_tmp - tsfsp_ctx->t_now_,
                tsfsp_ctx->nstep,
                tsfsp_ctx->njac);
  }
  tsfsp_ctx->t_now_ = t_now_tmp;

  if (tsfsp_ctx->logging_enabled)
  {
    tsfsp_ctx->perf_info.model_time[tsfsp_ctx->perf_info.n_step] = tsfsp_ctx->t_now_;
    ierr = VecGetSize(*tsfsp_ctx->solution_,&tsfsp_ctx->perf_info.n_eqs[size_t(tsfsp_ctx->perf_info.n_step)]);
    CHKERRQ(ierr);
    ierr = PetscTime(&tsfsp_ctx->perf_info.cpu_time[tsfsp_ctx->perf_info.n_step]);
    CHKERRQ(ierr);
    tsfsp_ctx->perf_info.n_step += 1;
  }

  return 0;
}

int pacmensl::TsFsp::TSRhsFunc(TS ts,PetscReal t,Vec u,Vec F,void *ctx)
{
  OdeSolverBase *ode_solver = ( OdeSolverBase * ) ctx;
  return ode_solver->EvaluateRHS(t,u,F);
}

int pacmensl::TsFsp::TSJacFunc(TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx)
{
  int  ierr;
  auto solver = ( TsFsp * ) ctx;

  ierr = solver->fspmat_->ComputeRHSJacobian(t,A);
  PACMENSLCHKERRQ(ierr);

  // Turn on update mode for subsequent Jacobian evaluations

  if (B != A)
  {
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  }

  solver->njac += 1;
  return 0;
}

// The implicit formulation is (-1)p_t + Ap = 0
int pacmensl::TsFsp::TSIFunc(TS ts,PetscReal t,Vec u,Vec u_t,Vec F,void *ctx)
{
  int           ierr;
  OdeSolverBase *ode_solver = ( OdeSolverBase * ) ctx;
  ierr = ode_solver->EvaluateRHS(t,u,F);
  PACMENSLCHKERRQ(ierr);
  ierr = VecAXPY(F,-1.0,u_t);
  PACMENSLCHKERRQ(ierr);
  return 0;
}

int pacmensl::TsFsp::TSIJacFunc(TS ts,PetscReal t,Vec u,Vec u_t,PetscReal a,Mat A,Mat P,void *ctx)
{
  int  ierr;
  auto solver = ( TsFsp * ) ctx;

  ierr = solver->fspmat_->ComputeRHSJacobian(t,A);
  PACMENSLCHKERRQ(ierr);

  // Turn on update mode for subsequent Jacobian evaluations

  ierr = MatShift(A,-1.0 * a);
  CHKERRQ(ierr);
  if (P != A)
  {
    MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);
  }

  solver->njac += 1;
  return 0;
}

int pacmensl::TsFsp::CheckImplicitType(TSType type, int* implicit)
{
  if (
      !strcmp(type, TSEULER) ||
      !strcmp(type, TSRK) ||
      !strcmp(type, TSEULER)
  )
  {
    *implicit = 0;
  }
  else{
    *implicit = 1;
  }
  return 0;
}

PacmenslErrorCode pacmensl::TsFsp::SetTsType(std::string type){
  type_ = std::move(type);
  return 0;
};
