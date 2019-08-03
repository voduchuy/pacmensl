//
// Created by Huy Vo on 6/4/19.
//

#include "DiscreteDistribution.h"

pacmensl::DiscreteDistribution::~DiscreteDistribution() {
  int ierr;
  if (p_ != nullptr) {
    ierr = VecDestroy(&p_);
    CHKERRABORT(MPI_COMM_SELF, ierr);
  }
  if (comm_ != nullptr) MPI_Comm_free(&comm_);
  p_    = nullptr;
  comm_ = nullptr;
}

pacmensl::DiscreteDistribution::DiscreteDistribution(const pacmensl::DiscreteDistribution &dist) {
  PetscErrorCode ierr;
  MPI_Comm_dup(dist.comm_, &comm_);
  t_   = dist.t_;
  if (p_ != PETSC_NULL) {
    ierr = VecDestroy(&p_); CHKERRABORT(comm_, ierr);
  }
  ierr = VecDuplicate(dist.p_, &p_); CHKERRABORT(comm_, ierr);
  ierr = VecCopy(dist.p_, p_); CHKERRABORT(comm_, ierr);
  states_ = dist.states_;
}

pacmensl::DiscreteDistribution::DiscreteDistribution() {
}

pacmensl::DiscreteDistribution::DiscreteDistribution(pacmensl::DiscreteDistribution &&dist) noexcept {
  comm_   = dist.comm_;
  t_      = dist.t_;
  states_ = std::move(dist.states_);
  p_      = dist.p_;

  dist.p_    = nullptr;
  dist.comm_ = nullptr;
}

pacmensl::DiscreteDistribution &
pacmensl::DiscreteDistribution::operator=(const pacmensl::DiscreteDistribution &dist) {
  PetscErrorCode ierr;

  if (comm_ != nullptr) {
    ierr = MPI_Comm_free(&comm_); MPICHKERRABORT(comm_, ierr);
  }
  if (p_ != PETSC_NULL) {
    ierr = VecDestroy(&p_); CHKERRABORT(comm_, ierr);
  }
  states_.clear();

  MPI_Comm_dup(dist.comm_, &comm_);
  t_ = dist.t_;
  VecDuplicate(dist.p_, &p_);
  VecCopy(dist.p_, p_);
  states_ = dist.states_;
  return *this;
}

pacmensl::DiscreteDistribution &
pacmensl::DiscreteDistribution::operator=(pacmensl::DiscreteDistribution &&dist) noexcept {
  if (this != &dist) {
    PetscErrorCode ierr;

    if (comm_ != nullptr) {
      ierr = MPI_Comm_free(&comm_); MPICHKERRABORT(comm_, ierr);
    }
    if (p_ != PETSC_NULL) {
      ierr = VecDestroy(&p_); CHKERRABORT(comm_, ierr);
    }
    states_.clear();

    comm_   = dist.comm_;
    t_      = dist.t_;
    states_ = std::move(dist.states_);
    p_      = dist.p_;

    dist.comm_ = nullptr;
    dist.p_    = nullptr;
    dist.states_.clear();

  }
  return *this;
}

pacmensl::DiscreteDistribution::DiscreteDistribution(MPI_Comm comm,
                                                     double t,
                                                     const pacmensl::StateSetBase *state_set,
                                                     const Vec &p) {
  PetscErrorCode ierr;
  MPI_Comm_dup(comm, &this->comm_);
  this->t_      = t;
  this->states_ = state_set->CopyStatesOnProc();
  ierr = VecDuplicate(p, &this->p_);
  ierr = VecCopy(p, this->p_);
}

int pacmensl::DiscreteDistribution::GetStateView(int &num_states, int &num_species, int *&states) {
  num_states  = states_.n_cols;
  num_species = states_.n_rows;
  states      = &states_[0];
  return 0;
}

int pacmensl::DiscreteDistribution::GetProbView(int &num_states, double *&p) {
  int ierr;
  ierr = VecGetLocalSize(p_, &num_states); CHKERRQ(ierr);
  ierr = VecGetArray(p_, &p); CHKERRQ(ierr);
  return 0;
}

int pacmensl::DiscreteDistribution::RestoreProbView(double *&p) {
  int ierr;
  if (p != nullptr) {
    ierr = VecRestoreArray(p_, &p); CHKERRQ(ierr);
  }
  return 0;
}

arma::Col<PetscReal> pacmensl::Compute1DMarginal(const DiscreteDistribution &dist, int species) {
  arma::Col<PetscReal> md_on_proc;
  // Find the max molecular count
  int                  num_species = dist.states_.n_rows;
  arma::Col<int>       max_molecular_counts_on_proc(num_species);
  arma::Col<int>       max_molecular_counts(num_species);
  max_molecular_counts_on_proc = arma::max(dist.states_, 1);
  int ierr = MPI_Allreduce(&max_molecular_counts_on_proc[0],
                           &max_molecular_counts[0],
                           num_species,
                           MPI_INT,
                           MPI_MAX,
                           dist.comm_); MPICHKERRABORT(dist.comm_, ierr);
  md_on_proc.resize(max_molecular_counts(species) + 1);
  md_on_proc.fill(0.0);
  PetscReal *p_dat;
  VecGetArray(dist.p_, &p_dat);
  for (int i{0}; i < dist.states_.n_cols; ++i) {
    md_on_proc(dist.states_(species, i)) += p_dat[i];
  }
  VecRestoreArray(dist.p_, &p_dat);
  arma::Col<PetscReal> md(md_on_proc.n_elem);
  MPI_Allreduce(( void * ) md_on_proc.memptr(),
                ( void * ) md.memptr(),
                md_on_proc.n_elem,
                MPIU_REAL,
                MPIU_SUM,
                dist.comm_);
  return md;
}
