//
// Created by Huy Vo on 6/4/19.
//

#include "DiscreteDistribution.h"

pecmeal::DiscreteDistribution::~DiscreteDistribution() {
  if (p != PETSC_NULL) VecDestroy(&p);
  if (comm_ != nullptr) MPI_Comm_free(&comm_);
  p = nullptr;
  comm_ = nullptr;
}

pecmeal::DiscreteDistribution::DiscreteDistribution(const pecmeal::DiscreteDistribution &dist) {
  PetscErrorCode ierr;
  MPI_Comm_dup(dist.comm_, &comm_);
  t_ = dist.t_;
  if (p != PETSC_NULL) {
    ierr = VecDestroy(&p);
    CHKERRABORT(comm_, ierr);
  }
  ierr = VecDuplicate(dist.p, &p);
  CHKERRABORT(comm_, ierr);
  ierr = VecCopy(dist.p, p);
  CHKERRABORT(comm_, ierr);
  states = dist.states;
}

pecmeal::DiscreteDistribution::DiscreteDistribution() {
}

pecmeal::DiscreteDistribution::DiscreteDistribution(pecmeal::DiscreteDistribution &&dist) noexcept {
  comm_ = dist.comm_;
  t_ = dist.t_;
  states = std::move(dist.states);
  p = dist.p;

  dist.p = nullptr;
  dist.comm_ = nullptr;
}

pecmeal::DiscreteDistribution &
pecmeal::DiscreteDistribution::operator=(const pecmeal::DiscreteDistribution &dist) {
  PetscErrorCode ierr;

  if (comm_ != nullptr) {
    ierr = MPI_Comm_free(&comm_);
    MPICHKERRABORT(comm_, ierr);
  }
  if (p != PETSC_NULL) {
    ierr = VecDestroy(&p);
    CHKERRABORT(comm_, ierr);
  }
  states.clear();

  MPI_Comm_dup(dist.comm_, &comm_);
  t_ = dist.t_;
  VecDuplicate(dist.p, &p);
  VecCopy(dist.p, p);
  states = dist.states;
  return *this;
}

pecmeal::DiscreteDistribution &
pecmeal::DiscreteDistribution::operator=(pecmeal::DiscreteDistribution &&dist) noexcept {
  PetscErrorCode ierr;

  if (comm_ != nullptr) {
    ierr = MPI_Comm_free(&comm_);
    MPICHKERRABORT(comm_, ierr);
  }
  if (p != PETSC_NULL) {
    ierr = VecDestroy(&p);
    CHKERRABORT(comm_, ierr);
  }
  states.clear();

  comm_ = dist.comm_;
  t_ = dist.t_;
  states = std::move(dist.states);
  p = dist.p;

  dist.comm_ = nullptr;
  dist.p = PETSC_NULL;
  dist.states.clear();

  return *this;
}
pecmeal::DiscreteDistribution::DiscreteDistribution(MPI_Comm comm,
                                                    double t,
                                                    const pecmeal::StateSetBase *state_set,
                                                    const Vec &p) {
  PetscErrorCode ierr;
  MPI_Comm_dup(comm, &this->comm_);
  this->t_ = t;
  this->states = state_set->CopyStatesOnProc();
  ierr = VecDuplicate(p, &this->p);
  CHKERRABORT(comm, ierr);
  ierr = VecCopy(p, this->p);
  CHKERRABORT(comm, ierr);
}

arma::Col<PetscReal> pecmeal::Compute1DMarginal(const pecmeal::DiscreteDistribution dist, int species) {
  arma::Col<PetscReal> md_on_proc;
  // Find the max molecular count
  int num_species = dist.states.n_rows;
  arma::Col<int> max_molecular_counts_on_proc(num_species);
  arma::Col<int> max_molecular_counts(num_species);
  max_molecular_counts_on_proc = arma::max(dist.states, 1);
  int ierr = MPI_Allreduce(&max_molecular_counts_on_proc[0],
                           &max_molecular_counts[0],
                           num_species,
                           MPI_INT,
                           MPI_MAX,
                           dist.comm_);
  MPICHKERRABORT(dist.comm_, ierr);
  md_on_proc.resize(max_molecular_counts(species) + 1);
  md_on_proc.fill(0.0);
  PetscReal *p_dat;
  VecGetArray(dist.p, &p_dat);
  for (int i{0}; i < dist.states.n_cols; ++i) {
    md_on_proc(dist.states(species, i)) += p_dat[i];
  }
  VecRestoreArray(dist.p, &p_dat);
  arma::Col<PetscReal> md(md_on_proc.n_elem);
  MPI_Allreduce((void *) md_on_proc.memptr(), (void *) md.memptr(), md_on_proc.n_elem, MPI_DOUBLE, MPI_SUM, dist.comm_);
  return md;
}
