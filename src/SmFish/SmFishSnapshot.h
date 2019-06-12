//
// Created by Huy Vo on 6/9/19.
//

#ifndef PECMEAL_SMFISHSNAPSHOT_H
#define PECMEAL_SMFISHSNAPSHOT_H

#include<armadillo>
#include"DiscreteDistribution.h"

namespace pecmeal {
class SmFishSnapshot {
 protected:
  arma::Mat<int> observations_;
  arma::Row<int> frequencies_;
  std::map<std::vector<int>, int> ob2ind;
  bool has_data_ = false;
  bool has_dictionary_ = false;
 public:
  SmFishSnapshot() = default;
  SmFishSnapshot(const arma::Mat<int> &observations, const arma::Row<int> &frequencies);

  SmFishSnapshot& operator= (SmFishSnapshot&& src) noexcept;

  void GenerateMap();

  int GetObservationIndex(const arma::Col<int> &x) const;
  int GetNumObservations() const;
  const arma::Row<int> &GetFrequencies() const;

  void Clear();
};

double SmFishSnapshotLogLikelihood(const SmFishSnapshot &data,
                                   const DiscreteDistribution &distribution,
                                   arma::Col<int> measured_species = arma::Col<int>({}),
                                   bool use_base_2 = false);
}

#endif //PECMEAL_SMFISHSNAPSHOT_H
