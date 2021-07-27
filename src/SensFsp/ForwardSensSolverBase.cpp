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
#include <PetscWrap/PetscWrap.h>
#include "ForwardSensSolverBase.h"

pacmensl::ForwardSensSolverBase::ForwardSensSolverBase(MPI_Comm new_comm) {
  int ierr;
  comm_ = new_comm;
  ierr = MPI_Comm_rank(comm_, &my_rank_);
  MPICHKERRTHROW(ierr);
  ierr = MPI_Comm_size(comm_, &comm_size_);
  MPICHKERRTHROW(ierr);
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetInitialSolution(pacmensl::Petsc<Vec> &sol) {
  if (sol == nullptr){
    return -1;
  }
  solution_ = sol.mem();
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetInitialSensitivity(std::vector<Petsc < Vec>> &sens_vecs) {
  num_parameters_ = sens_vecs.size();
  sens_vecs_.resize(num_parameters_);
  for (auto i=0; i < num_parameters_; ++i){
    sens_vecs_[i] = sens_vecs[i].mem();
  }
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetRhs(pacmensl::ForwardSensSolverBase::RhsFun rhs) {
  rhs_ = rhs;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetSensRhs(pacmensl::ForwardSensSolverBase::SensRhs1Fun sensrhs) {
  srhs_ = sensrhs;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetStopCondition(const std::function<int(PetscReal,
                                                                                            Vec,
                                                                                            int,
                                                                                            Vec *,
                                                                                            void *)> &stop_check,
                                                                    void *stop_data_) {
  stop_check_ = stop_check;
  stop_data_ = stop_data_;
  return 0;
}

pacmensl::ForwardSensSolverBase::~ForwardSensSolverBase() {
  comm_ = MPI_COMM_NULL;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::FreeWorkspace() {
  solution_ = nullptr;
  sens_vecs_.clear();
  t_now_ = 0.0;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetFinalTime(PetscReal _t_final) {
  t_final_ = _t_final;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetCurrentTime(PetscReal t) {
  t_now_ = t;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetStatusOutput(int iprint) {
  print_intermediate = iprint;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::EnableLogging() {
  logging_enabled = PETSC_TRUE;
  perf_info.n_step = 0;
  perf_info.model_time.resize(100000);
  perf_info.cpu_time.resize(100000);
  perf_info.n_eqs.resize(100000);
  return 0;
}
