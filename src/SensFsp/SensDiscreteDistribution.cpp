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

#include "SensDiscreteDistribution.h"

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution() : DiscreteDistribution()
{
}

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution(MPI_Comm comm,
                                                             double t,
                                                             const pacmensl::StateSetBase *state_set,
                                                             const Vec &p,
                                                             const std::vector<Vec> &dp) : DiscreteDistribution(comm,
                                                                                                                t,
                                                                                                                state_set,
                                                                                                                p)
{
  PacmenslErrorCode ierr;
  dp_.resize(dp.size());
  for (int i{0}; i < dp_.size(); ++i)
  {
    ierr = VecDuplicate(dp[i], &dp_[i]);
    PACMENSLCHKERRTHROW(ierr);
    ierr = VecCopy(dp.at(i), dp_.at(i));
    PACMENSLCHKERRTHROW(ierr);
  }
}

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution(const pacmensl::SensDiscreteDistribution &dist)
    : DiscreteDistribution(( const pacmensl::DiscreteDistribution & ) dist)
{
  PacmenslErrorCode ierr;
  dp_.resize(dist.dp_.size());
  for (int i{0}; i < dp_.size(); ++i)
  {
    VecDuplicate(dist.dp_[i], &dp_[i]);
    VecCopy(dist.dp_[i], dp_[i]);
  }
}

pacmensl::SensDiscreteDistribution::SensDiscreteDistribution(pacmensl::SensDiscreteDistribution &&dist) noexcept
    : DiscreteDistribution(( pacmensl::DiscreteDistribution && ) dist)
{
  dp_ = std::move(dist.dp_);
}

pacmensl::SensDiscreteDistribution &pacmensl::SensDiscreteDistribution::operator=(const pacmensl::SensDiscreteDistribution &dist)
{
  DiscreteDistribution::operator=(( const DiscreteDistribution & ) dist);

  for (int i{0}; i < dp_.size(); ++i)
  {
    VecDestroy(&dp_[i]);
  }
  dp_.resize(dist.dp_.size());
  for (int i{0}; i < dp_.size(); ++i)
  {
    VecDuplicate(dist.dp_[i], &dp_[i]);
    VecCopy(dist.dp_[i], dp_[i]);
  }
  return *this;
}

pacmensl::SensDiscreteDistribution &pacmensl::SensDiscreteDistribution::operator=(pacmensl::SensDiscreteDistribution &&dist) noexcept
{
  if (this != &dist)
  {
    DiscreteDistribution::operator=(( DiscreteDistribution && ) dist);
    for (int i{0}; i < dp_.size(); ++i)
    {
      VecDestroy(&dp_[i]);
    }
    dp_ = std::move(dist.dp_);
  }
  return *this;
}

PacmenslErrorCode pacmensl::SensDiscreteDistribution::GetSensView(int is, int &num_states, double *&p)
{
  int ierr;
  if (is >= dp_.size()) return -1;
  ierr = VecGetLocalSize(dp_[is], &num_states);
  CHKERRQ(ierr);
  ierr = VecGetArray(dp_[is], &p);
  CHKERRQ(ierr);
  return 0;
}

PacmenslErrorCode pacmensl::SensDiscreteDistribution::RestoreSensView(int is, double *&p)
{
  PacmenslErrorCode ierr;
  if (is >= dp_.size()) return -1;
  if (p != nullptr)
  {
    ierr = VecRestoreArray(dp_[is], &p);
    CHKERRQ(ierr);
  }
  return 0;
}

pacmensl::SensDiscreteDistribution::~SensDiscreteDistribution()
{
  for (int i{0}; i < dp_.size(); ++i)
  {
    VecDestroy(&dp_[i]);
  }
}

PacmenslErrorCode pacmensl::SensDiscreteDistribution::WeightedAverage(int is, int nout,
                                                                      PetscReal *fout,
                                                                      std::function<PacmenslErrorCode(int,
                                                                                                      int *,
                                                                                                      int,
                                                                                                      PetscReal *,
                                                                                                      void *)> weight_func,
                                                                      void *wf_args)
{
  PacmenslErrorCode ierr;

  int num_local_states;
  PetscReal* plocal;
  if (is >= 0){
    ierr = GetSensView(is, num_local_states, plocal); PACMENSLCHKERRQ(ierr);
  }
  else{
    ierr = GetProbView(num_local_states, plocal); PACMENSLCHKERRQ(ierr);
  }

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

PacmenslErrorCode pacmensl::Compute1DSensMarginal(const pacmensl::SensDiscreteDistribution &dist,
                                                  int is,
                                                  int species,
                                                  arma::Col<PetscReal> &out)
{
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
  for (int i{0}; i < dist.states_.n_cols; ++i)
  {
    md_on_proc(dist.states_(species, i)) += dp_dat[i];
  }
  VecRestoreArray(dist.dp_[is], &dp_dat);

  if (out.is_empty()){
    out.set_size(md_on_proc.size());
  }
  MPI_Allreduce(( void * ) md_on_proc.memptr(),
                ( void * ) out.memptr(),
                md_on_proc.n_elem,
                MPIU_REAL,
                MPIU_SUM,
                dist.comm_);

  return 0;
}

PacmenslErrorCode pacmensl::ComputeFIM(SensDiscreteDistribution &dist, arma::Mat<PetscReal> &fim)
{
  int ierr;
  if (!fim.is_empty() && (fim.n_rows != dist.dp_.size() || fim.n_cols != dist.dp_.size())) return -1;
  if (fim.is_empty())
  {
    fim.set_size(dist.dp_.size(), dist.dp_.size());
  }
  bool     warn{false};
  int      num_par = dist.dp_.size();
  for (int i       = 0; i < num_par; ++i)
  {
    for (int j{0}; j <= i; ++j)
    {
      fim(i, j) = 0.0;
      int num_states;
      PetscReal *si;
      PetscReal *sj;
      PetscReal *p;
      ierr = dist.GetProbView(num_states, p); PACMENSLCHKERRQ(ierr);
      ierr = dist.GetSensView(i, num_states, si); PACMENSLCHKERRQ(ierr);
      if (i!=j)
      {
        ierr = dist.GetSensView(j, num_states, sj); PACMENSLCHKERRQ(ierr);
      }
      else{
        sj = si;
      }
      for (int k{0}; k < num_states; ++k)
      {
        if (p[k] < 1.0e-16)
        {
          p[k] = 1.0e-16;
          warn = true;
        }
        fim(i, j) += si[k] * sj[k] / p[k];
      }
      dist.RestoreProbView(p);
      dist.RestoreSensView(i, si);
      if (i!=j)
      {
        dist.RestoreSensView(j, sj);
      }
      PetscReal tmp;
      ierr = MPI_Allreduce((void*)(fim.colptr(j) + i), &tmp, 1, MPIU_REAL, MPIU_SUM, dist.comm_); PACMENSLCHKERRQ(ierr);
      fim(i,j) = tmp;
    }
  }
  for (int i = 0; i < num_par; ++i){
    for (int j{i+1}; j < num_par; ++j){
      fim(i, j) = fim(j, i);
    }
  }
  if (warn) PetscPrintf(dist.comm_, "Warning: rounding was done in FIM computation.\n");
  return 0;
}
