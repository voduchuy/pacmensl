//
// Created by Huy Vo on 2019-06-28.
//

#include "ForwardSensSolverBase.h"

pacmensl::ForwardSensSolverBase::ForwardSensSolverBase(MPI_Comm new_comm) {
  int ierr;
  ierr = MPI_Comm_dup(new_comm, &comm_);
  MPICHKERRTHROW(ierr);
  ierr = MPI_Comm_rank(comm_, &my_rank_);
  MPICHKERRTHROW(ierr);
  ierr = MPI_Comm_size(comm_, &comm_size_);
  MPICHKERRTHROW(ierr);
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetInitialSolution(Vec &sol) {
  if (sol == nullptr){
    return -1;
  }
  solution_ = &sol;
  return 0;
}

PacmenslErrorCode pacmensl::ForwardSensSolverBase::SetInitialSensitivity(std::vector<Vec> &sens_vecs) {
  num_parameters_ = sens_vecs.size();
  sens_vecs_.resize(num_parameters_);
  for (auto i=0; i < num_parameters_; ++i){
    sens_vecs_[i] = &sens_vecs[i];
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
  if (comm_ != nullptr){
    MPI_Comm_free(&comm_);
  }
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
