//
// Created by Huy Vo on 2019-06-28.
//

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
