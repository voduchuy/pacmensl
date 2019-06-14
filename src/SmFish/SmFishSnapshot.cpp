//
// Created by Huy Vo on 6/9/19.
//

#include "SmFishSnapshot.h"

pecmeal::SmFishSnapshot::SmFishSnapshot(const arma::Mat<int> &observations, const arma::Row<int> &frequencies) {
  observations_ = observations;
  frequencies_ = frequencies;
  has_data_ = true;
  GenerateMap();
}

void pecmeal::SmFishSnapshot::GenerateMap() {
  if (!has_data_) {
    std::cout << "pecmeal::SmFishSnapshot: Cannot generate a map without data.\n";
    return;
  }

  if (has_dictionary_) return;

  int num_observations = observations_.n_cols;
  for (int i{0}; i < num_observations; ++i) {
    ob2ind.insert(std::pair<std::vector<int>, int>(arma::conv_to<std::vector<int>>::from(observations_.col(i)), i));
  }
  has_dictionary_ = true;
}

void pecmeal::SmFishSnapshot::Clear() {
  observations_.clear();
  frequencies_.clear();
  ob2ind.clear();
  has_dictionary_ = false;
  has_data_ = false;
}

int pecmeal::SmFishSnapshot::GetObservationIndex(const arma::Col<int> &x) const {
  if (x.n_elem != observations_.n_rows) {
    return -1;
  }
  auto i = ob2ind.find(arma::conv_to<std::vector<int>>::from(x));
  if (i != ob2ind.end()) {
    return i->second;
  } else {
    return -1;
  }
}

int pecmeal::SmFishSnapshot::GetNumObservations() const {
  return observations_.n_cols;
}
const arma::Row<int> &pecmeal::SmFishSnapshot::GetFrequencies() const {
  return frequencies_;
}

pecmeal::SmFishSnapshot &pecmeal::SmFishSnapshot::operator=(pecmeal::SmFishSnapshot &&src) noexcept {
  Clear();

  observations_ = std::move(src.observations_);
  frequencies_ = std::move(src.frequencies_);
  has_data_ = true;
  GenerateMap();

  src.Clear();
  return *this;
}

double pecmeal::SmFishSnapshotLogLikelihood(const SmFishSnapshot &data,
                                            const DiscreteDistribution &distribution,
                                            arma::Col<int> measured_species,
                                            bool use_base_2) {

  int ierr;

  if (measured_species.empty()) {
    measured_species = arma::regspace<arma::Col<int>>(0, distribution.states.n_rows - 1);
  }

  MPI_Comm comm = distribution.comm_;
  int num_observations = data.GetNumObservations();

  const PetscReal *p_dat;
  VecGetArrayRead(distribution.p, &p_dat);

  arma::Col<double> predicted_probabilities_local(num_observations, arma::fill::zeros);
  arma::Col<double> predicted_probabilities = predicted_probabilities_local;

  for (int i{0}; i < distribution.states.n_cols; ++i) {
    arma::Col<int> x(measured_species.n_elem);
    for (int j = 0; j < measured_species.n_elem; ++j) {
      x(j) = distribution.states(measured_species(j), i);
    }
    int k = data.GetObservationIndex(x);
    if (k != -1) predicted_probabilities_local(k) += p_dat[i];
  }
  VecRestoreArrayRead(distribution.p, &p_dat);
  ierr = MPI_Allreduce(&predicted_probabilities_local[0],
                       &predicted_probabilities[0],
                       num_observations,
                       MPI_DOUBLE,
                       MPI_SUM,
                       comm);
  MPICHKERRABORT(comm, ierr);

  const arma::Row<int> &freq = data.GetFrequencies();

  double ll = 0.0;
  for (int i{0}; i < num_observations; ++i) {
    if (!use_base_2) {
      ll += freq(i) * log(std::max(1.0e-16, predicted_probabilities(i)));
    } else {
      ll += freq(i) * log2(std::max(1.0e-16, predicted_probabilities(i)));
    }
  }

  return ll;
}
