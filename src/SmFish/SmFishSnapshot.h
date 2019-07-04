//
// Created by Huy Vo on 6/9/19.
//

#ifndef PACMENSL_SMFISHSNAPSHOT_H
#define PACMENSL_SMFISHSNAPSHOT_H

#include<armadillo>
#include"DiscreteDistribution.h"
#include<map>

namespace pacmensl {
class SmFishSnapshot
{
 protected:
  arma::Mat<int>                  observations_;
  arma::Row<int>                  frequencies_;
  std::map<std::vector<int>, int> ob2ind;
  bool                            has_data_       = false;
  bool                            has_dictionary_ = false;
  void GenerateMap();
 public:
  SmFishSnapshot() = default;
  SmFishSnapshot(const arma::Mat<int> &observations);
  SmFishSnapshot(const arma::Mat<int> &observations, const arma::Row<int> &frequencies);
  SmFishSnapshot &operator=(SmFishSnapshot &&src) noexcept;
  int GetObservationIndex(const arma::Col<int> &x) const;
  int GetNumObservations() const;
  const arma::Mat<int> &GetObservations() const;
  const arma::Row<int> &GetFrequencies() const;

  void Clear();
};

double SmFishSnapshotLogLikelihood(const SmFishSnapshot &data,
                                   const DiscreteDistribution &distribution,
                                   arma::Col<int> measured_species = arma::Col<int>({}),
                                   bool use_base_2 = false);
}

#endif //PACMENSL_SMFISHSNAPSHOT_H
