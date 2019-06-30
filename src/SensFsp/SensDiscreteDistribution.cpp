//
// Created by Huy Vo on 2019-06-28.
//

#include "SensDiscreteDistribution.h"

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution() : DiscreteDistribution() {
}

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution(MPI_Comm comm,
                                                             double t,
                                                             const pacmensl::StateSetBase *state_set,
                                                             const Vec &p,
                                                             const std::vector<Vec> &dp) : DiscreteDistribution(comm,
                                                                                                                t,
                                                                                                                state_set,
                                                                                                                p) {
  PacmenslErrorCode ierr;
  dp_.resize(dp.size());
  for (int i{0}; i < dp_.size(); ++i) {
    ierr = VecDuplicate(dp[i], &dp_[i]);
    PACMENSLCHKERRTHROW(ierr);
    ierr = VecCopy(dp.at(i), dp_.at(i));
    PACMENSLCHKERRTHROW(ierr);
  }
}

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution(const pacmensl::SensDiscreteDistribution &dist)
    : DiscreteDistribution(( const pacmensl::DiscreteDistribution & ) dist) {
  PacmenslErrorCode ierr;
  dp_.resize(dist.dp_.size());
  for (int i{0}; i < dp_.size(); ++i) {
    VecDuplicate(dist.dp_[i], &dp_[i]);
    VecCopy(dist.dp_[i], dp_[i]);
  }
}

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution(pacmensl::SensDiscreteDistribution &&dist) noexcept
    : DiscreteDistribution(( pacmensl::DiscreteDistribution && ) dist) {
  dp_ = std::move(dist.dp_);
}

pacmensl::SensDiscreteDistribution &pacmensl::SensDiscreteDistribution::operator=(const pacmensl::SensDiscreteDistribution &dist) {
  DiscreteDistribution::operator=(( const DiscreteDistribution & ) dist);

  for (int i{0}; i < dp_.size(); ++i) {
    VecDestroy(&dp_[i]);
  }
  dp_.resize(dist.dp_.size());
  for (int i{0}; i < dp_.size(); ++i) {
    VecDuplicate(dist.dp_[i], &dp_[i]);
    VecCopy(dist.dp_[i], dp_[i]);
  }
  return *this;
}

pacmensl::SensDiscreteDistribution &pacmensl::SensDiscreteDistribution::operator=(pacmensl::SensDiscreteDistribution &&dist) noexcept {
  DiscreteDistribution::operator=(( DiscreteDistribution && ) dist);
  for (int i{0}; i < dp_.size(); ++i) {
    VecDestroy(&dp_[i]);
  }
  dp_ = std::move(dist.dp_);
  return *this;
}

PacmenslErrorCode pacmensl::SensDiscreteDistribution::GetSensView(int is, int num_states, double *&p) {
  int ierr;
  if (is >= dp_.size()) return -1;
  ierr = VecGetSize(dp_[is], &num_states);
  CHKERRQ(ierr);
  ierr = VecGetArray(dp_[is], &p);
  CHKERRQ(ierr);
  return 0;
}

PacmenslErrorCode pacmensl::SensDiscreteDistribution::RestoreSensView(int is, int num_states, double *&p) {
  PacmenslErrorCode ierr;
  if (is >= dp_.size()) return -1;
  if (p != nullptr) {
    ierr = VecRestoreArray(dp_[is], &p);
    CHKERRQ(ierr);
  }
  return 0;
}

pacmensl::SensDiscreteDistribution::~SensDiscreteDistribution() {
  for (int i{0}; i < dp_.size(); ++i) {
    VecDestroy(&dp_[i]);
  }
}

arma::Col<PetscReal> pacmensl::Compute1DSensMarginal(const pacmensl::SensDiscreteDistribution& dist, int is, int species) {
  if (is > dist.dp_.size()) PACMENSLCHKERRTHROW(-1);
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
                           dist.comm_);
  PACMENSLCHKERRTHROW(ierr);
  md_on_proc.resize(max_molecular_counts(species) + 1);
  md_on_proc.fill(0.0);
  PetscReal *dp_dat;
  VecGetArray(dist.dp_[is], &dp_dat);
  for (int i{0}; i < dist.states_.n_cols; ++i) {
    md_on_proc(dist.states_(species, i)) += dp_dat[i];
  }
  VecRestoreArray(dist.dp_[is], &dp_dat);
  arma::Col<PetscReal> md(md_on_proc.n_elem);
  MPI_Allreduce(( void * ) md_on_proc.memptr(),
                ( void * ) md.memptr(),
                md_on_proc.n_elem,
                MPIU_REAL,
                MPIU_SUM,
                dist.comm_);
  return md;
}
