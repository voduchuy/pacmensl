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
#include "CvodeFsp.h"
#include "OdeSolver/OdeSolverBase.h"
#include <OdeSolver/CvodeFsp.h>
#include "OdeSolverBase.h"

int pacmensl::OdeSolverBase::SetTolerances(PetscReal _r_tol, PetscReal _abs_tol) {
  rel_tol_ = _r_tol;
  abs_tol_ = _abs_tol;
  return 0;
}

namespace pacmensl {

int OdeSolverBase::SetStatusOutput(int iprint) {
  print_intermediate = iprint;
  return 0;
}

OdeSolverBase::OdeSolverBase(MPI_Comm new_comm) {
  int ierr;
  comm_ = new_comm;
  ierr = MPI_Comm_rank(comm_, &my_rank_); PACMENSLCHKERRTHROW(ierr);
  ierr = MPI_Comm_size(comm_, &comm_size_); PACMENSLCHKERRTHROW(ierr);
}

int OdeSolverBase::SetFinalTime(PetscReal _t_final) {
  t_final_ = _t_final;
  return 0;
}

int OdeSolverBase::SetInitialSolution(Vec *_sol) {
  if (!_sol){
    PACMENSLCHKERRQ(-1);
  }
  solution_ = _sol;
  return 0;
}

PacmenslErrorCode OdeSolverBase::SetRhs(std::function<PacmenslErrorCode(PetscReal,Vec,Vec)> _rhs)
{
  int ierr{0};
  try{
    rhs_ = std::move(_rhs);
  }
  catch(...){
    ierr = -1;
  }
  PACMENSLCHKERRQ(ierr);

  return 0;
}

int OdeSolverBase::EvaluateRHS(PetscReal t, Vec x, Vec y) {
  PACMENSLCHKERRQ(rhs_(t, x, y));
  return 0;
}

int OdeSolverBase::SetCurrentTime(PetscReal t) {
  if (isnan(t)){
    printf("\n Time variable cannot have NaN value!\n");
    PACMENSLCHKERRQ(-1);
  }
  t_now_ = t;
  return 0;
}

PetscReal OdeSolverBase::GetCurrentTime() const {
  return t_now_;
}

PetscInt OdeSolverBase::Solve() {
  // Make sure the necessary data has been set
  if (solution_ == nullptr) return -1;
  if (rhs_ == nullptr) return -1;
  return 0;
}

OdeSolverBase::~OdeSolverBase() {
  comm_ = MPI_COMM_NULL;
}

int OdeSolverBase::EnableLogging() {
  logging_enabled = PETSC_TRUE;
  perf_info.n_step = 0;
  perf_info.model_time.resize(100000);
  perf_info.cpu_time.resize(100000);
  perf_info.n_eqs.resize(100000);
  return 0;
}

FiniteProblemSolverPerfInfo OdeSolverBase::GetAvgPerfInfo() const {
  assert(logging_enabled);

  FiniteProblemSolverPerfInfo perf_out = perf_info;

  PetscMPIInt comm_size;
  MPI_Comm_size(comm_, &comm_size);

  for (auto i{perf_out.n_step - 1}; i >= 0; --i) {
    perf_out.cpu_time[i] = perf_out.cpu_time[i] - perf_out.cpu_time[0];
    MPI_Allreduce(MPI_IN_PLACE, (void *) &perf_out.cpu_time[i], 1, MPIU_REAL, MPIU_SUM, comm_);
  }

  for (auto i{0}; i < perf_out.n_step; ++i) {
    perf_out.cpu_time[i] /= PetscReal(comm_size);
  }

  return perf_out;
}

int
OdeSolverBase::SetStopCondition(const std::function<PacmenslErrorCode (PetscReal, Vec, PetscReal&, void *)> &stop_check_, void *stop_data_) {
  OdeSolverBase::stop_check_ = stop_check_;
  OdeSolverBase::stop_data_ = stop_data_;
  return 0;
}

PacmenslErrorCode OdeSolverBase::SetFspMatPtr(FspMatrixBase* mat)
{
  fspmat_ = mat;
  return 0;
}
}