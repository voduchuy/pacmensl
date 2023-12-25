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

#include "DiscreteDistribution.h"

pacmensl::DiscreteDistribution::~DiscreteDistribution() {
  int ierr;
  if (p_ != nullptr) {
    ierr = VecDestroy(&p_);
    CHKERRABORT(MPI_COMM_SELF, ierr);
  }
  p_    = nullptr;
  comm_ = MPI_COMM_NULL;
}

pacmensl::DiscreteDistribution::DiscreteDistribution(const pacmensl::DiscreteDistribution &dist) {
  PetscErrorCode ierr;
  comm_ = dist.comm_;
  t_   = dist.t_;
  if (p_ != NULL) {
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
  dist.comm_ = MPI_COMM_NULL;
}

pacmensl::DiscreteDistribution &
pacmensl::DiscreteDistribution::operator=(const pacmensl::DiscreteDistribution &dist) {
  PetscErrorCode ierr;

  if (p_ != NULL && p_ != nullptr) {
    ierr = VecDestroy(&p_); CHKERRABORT(comm_, ierr);
  }
  states_.clear();

  comm_ = dist.comm_;
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

    if (p_ != NULL && p_ != nullptr) {
      ierr = VecDestroy(&p_); CHKERRABORT(comm_, ierr);
    }
    states_.clear();

    comm_   = dist.comm_;
    t_      = dist.t_;
    states_ = std::move(dist.states_);
    p_      = dist.p_;

    dist.comm_ = MPI_COMM_NULL;
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
  this->comm_ = comm;
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

PacmenslErrorCode pacmensl::DiscreteDistribution::WeightedAverage(int nout,
                                                                  PetscReal *fout,
                                                                  std::function<PacmenslErrorCode(int num_species, int *x,
                                                                                                  int nout, PetscReal *fx,
                                                                                                  void *args)> weight_func,
                                                                  void *wf_args)
{
  PacmenslErrorCode ierr;

  int num_local_states;
  PetscReal* plocal;
  ierr = GetProbView(num_local_states, plocal); PACMENSLCHKERRQ(ierr);

  for (int i = 0; i < nout; ++i)
  {
    fout[i] = 0.0;
  }

  PetscReal* wtmp = new PetscReal[nout];
  for (int j = 0; j < num_local_states; ++j)
  {
    ierr = weight_func(states_.n_rows, states_.colptr(j), nout, wtmp, wf_args); PACMENSLCHKERRQ(ierr);
    for (int i = 0; i < nout; ++i)
    {
      fout[i] += wtmp[i]*plocal[j];
    }
  }

  ierr = MPI_Allreduce(MPI_IN_PLACE, fout, nout, MPIU_REAL, MPIU_SUM, comm_);
  PACMENSLCHKERRQ(ierr);

  delete[] wtmp;
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
