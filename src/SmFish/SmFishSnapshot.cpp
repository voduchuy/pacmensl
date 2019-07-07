//
// Created by Huy Vo on 6/9/19.
//

#include "SmFishSnapshot.h"

pacmensl::SmFishSnapshot::SmFishSnapshot(const arma::Mat<int> &observations, const arma::Row<int> &frequencies)
{
  int unique_count = 0;
  observations_.set_size(observations.n_rows, observations.n_cols);
  frequencies_.set_size(observations.n_cols);
  frequencies_.fill(0);
  auto it = ob2ind.begin();
  for (int i{0}; i < observations.n_cols; ++i){
    it = ob2ind.find(arma::conv_to<std::vector<int>>::from(observations.col(i)));
    if (it != ob2ind.end()){
      frequencies_((*it).second) ++;
    }
    else{
      ob2ind.emplace(std::make_pair(arma::conv_to<std::vector<int>>::from(observations.col(i)), unique_count));
      observations_.col(unique_count) = observations.col(i);
      frequencies_(unique_count)+= frequencies(i);
      unique_count++;
    }
  }
  observations_.resize(observations.n_rows, unique_count);
  frequencies_.resize(unique_count);
  has_data_       = true;
  has_dictionary_ = true;
}

pacmensl::SmFishSnapshot::SmFishSnapshot(const arma::Mat<int> &observations)
{
  int unique_count = 0;
  observations_.set_size(observations.n_rows, observations.n_cols);
  frequencies_.set_size(observations.n_cols);
  frequencies_.fill(0);
  auto it = ob2ind.begin();
  for (int i{0}; i < observations.n_cols; ++i){
    it = ob2ind.find(arma::conv_to<std::vector<int>>::from(observations.col(i)));
    if (it != ob2ind.end()){
      frequencies_((*it).second) ++;
    }
    else{
      ob2ind.emplace(std::make_pair(arma::conv_to<std::vector<int>>::from(observations.col(i)), unique_count));
      observations_.col(unique_count) = observations.col(i);
      frequencies_(unique_count)++;
      unique_count++;
    }
  }
  observations_.resize(observations.n_rows, unique_count);
  frequencies_.resize(unique_count);
  has_data_       = true;
  has_dictionary_ = true;
}

void pacmensl::SmFishSnapshot::GenerateMap()
{
  if (!has_data_)
  {
    std::cout << "pacmensl::SmFishSnapshot: Cannot generate a map without data.\n";
    return;
  }

  if (has_dictionary_) return;

  int      num_observations = observations_.n_cols;
  for (int i{0}; i < num_observations; ++i)
  {
    ob2ind.insert(std::pair<std::vector<int>, int>(arma::conv_to<std::vector<int>>::from(observations_.col(i)), i));
  }
  has_dictionary_ = true;
}

void pacmensl::SmFishSnapshot::Clear()
{
  observations_.clear();
  frequencies_.clear();
  ob2ind.clear();
  has_dictionary_ = false;
  has_data_       = false;
}

int pacmensl::SmFishSnapshot::GetObservationIndex(const arma::Col<int> &x) const
{
  if (x.n_elem != observations_.n_rows)
  {
    return -1;
  }
  auto i = ob2ind.find(arma::conv_to<std::vector<int>>::from(x));
  if (i != ob2ind.end())
  {
    return i->second;
  } else
  {
    return -1;
  }
}

int pacmensl::SmFishSnapshot::GetNumObservations() const
{
  return observations_.n_cols;
}

const arma::Mat<int> &pacmensl::SmFishSnapshot::GetObservations() const
{
  return observations_;
}

const arma::Row<int> &pacmensl::SmFishSnapshot::GetFrequencies() const
{
  return frequencies_;
}

pacmensl::SmFishSnapshot &pacmensl::SmFishSnapshot::operator=(pacmensl::SmFishSnapshot &&src) noexcept
{
  Clear();

  observations_ = std::move(src.observations_);
  frequencies_  = std::move(src.frequencies_);
  has_data_     = true;
  GenerateMap();

  src.Clear();
  return *this;
}

double pacmensl::SmFishSnapshotLogLikelihood(const SmFishSnapshot &data,
                                             const DiscreteDistribution &distribution,
                                             arma::Col<int> measured_species,
                                             bool use_base_2)
{

  int ierr;

  if (measured_species.empty())
  {
    measured_species = arma::regspace<arma::Col<int>>(0, distribution.states_.n_rows - 1);
  }

  MPI_Comm comm                             = distribution.comm_;
  int      num_observations                 = data.GetNumObservations();

  const PetscReal *p_dat;
  VecGetArrayRead(distribution.p_, &p_dat);

  arma::Col<double> predicted_probabilities_local(num_observations, arma::fill::zeros);
  arma::Col<double> predicted_probabilities = predicted_probabilities_local;

  for (int i{0}; i < distribution.states_.n_cols; ++i)
  {
    arma::Col<int> x(measured_species.n_elem);
    for (int       j = 0; j < measured_species.n_elem; ++j)
    {
      x(j) = distribution.states_(measured_species(j), i);
    }
    int            k = data.GetObservationIndex(x);
    if (k != -1) predicted_probabilities_local(k) += p_dat[i];
  }
  VecRestoreArrayRead(distribution.p_, &p_dat);
  ierr = MPI_Allreduce(&predicted_probabilities_local[0],
                       &predicted_probabilities[0],
                       num_observations,
                       MPIU_REAL,
                       MPIU_SUM,
                       comm); MPICHKERRABORT(comm, ierr);

  const arma::Row<int> &freq = data.GetFrequencies();

  double   ll = 0.0;
  for (int i{0}; i < num_observations; ++i)
  {
    if (!use_base_2)
    {
      ll += freq(i) * log(std::max(1.0e-16, predicted_probabilities(i)));
    } else
    {
      ll += freq(i) * log2(std::max(1.0e-16, predicted_probabilities(i)));
    }
  }

  return ll;
}

PacmenslErrorCode pacmensl::SmFishSnapshotGradient(const pacmensl::SmFishSnapshot &data,
                                                   const pacmensl::SensDiscreteDistribution &distribution,
                                                   std::vector<PetscReal> &gradient,
                                                   arma::Col<int> measured_species,
                                                   bool use_base_2)
{
  PacmenslErrorCode ierr;

  int num_parameters = distribution.dp_.size();
  if (gradient.empty()){
    gradient.resize(num_parameters);
  }

  if (measured_species.empty())
  {
    measured_species = arma::regspace<arma::Col<int>>(0, distribution.states_.n_rows - 1);
  }

  MPI_Comm comm                             = distribution.comm_;
  int      num_observations                 = data.GetNumObservations();

  const arma::Row<int> &freq = data.GetFrequencies();

  const PetscReal *p_dat;
  VecGetArrayRead(distribution.p_, &p_dat);

  arma::Col<double> predicted_probabilities_local(num_observations, arma::fill::zeros);
  arma::Col<double> predicted_sensitivities_local(num_observations, arma::fill::zeros);
  arma::Col<double> predicted_probabilities = predicted_probabilities_local;
  arma::Col<double> predicted_sensitivities = predicted_sensitivities_local;

  // Compute the probabilities of observations
  for (int i{0}; i < distribution.states_.n_cols; ++i)
  {
    arma::Col<int> x(measured_species.n_elem);
    for (int       j = 0; j < measured_species.n_elem; ++j)
    {
      x(j) = distribution.states_(measured_species(j), i);
    }
    int            k = data.GetObservationIndex(x);
    if (k != -1) predicted_probabilities_local(k) += p_dat[i];
  }
  VecRestoreArrayRead(distribution.p_, &p_dat);
  ierr = MPI_Allreduce(&predicted_probabilities_local[0],
                       &predicted_probabilities[0],
                       num_observations,
                       MPIU_REAL,
                       MPIU_SUM,
                       comm); CHKERRMPI(ierr);

  // Compute the gradient
  int ns;
  for (int par{0}; par < num_parameters; ++par){
    ierr = VecGetArrayRead(distribution.dp_[par], &p_dat); PACMENSLCHKERRQ(ierr);
    for (int i{0}; i < distribution.states_.n_cols; ++i)
    {
      arma::Col<int> x(measured_species.n_elem);
      for (int       j = 0; j < measured_species.n_elem; ++j)
      {
        x(j) = distribution.states_(measured_species(j), i);
      }
      int            k = data.GetObservationIndex(x);
      if (k != -1) predicted_sensitivities_local(k) += p_dat[i];
    }
    ierr = VecRestoreArrayRead(distribution.dp_[par], &p_dat); PACMENSLCHKERRQ(ierr);
    ierr = MPI_Allreduce(&predicted_sensitivities_local[0],
                         &predicted_sensitivities[0],
                         num_observations,
                         MPIU_REAL,
                         MPIU_SUM,
                         comm); CHKERRMPI(ierr);

    gradient[par] = 0.0;
    for (int i{0}; i < num_observations; ++i)
    {
      if (!use_base_2)
      {
        gradient[par] += freq(i) * predicted_sensitivities(i)/(std::max(1.0e-16, predicted_probabilities(i)));
      } else
      {
        gradient[par] += freq(i) * predicted_sensitivities(i)/(log(2)*(std::max(1.0e-16, predicted_probabilities(i))));
      }
    }
  }
  return 0;
}
