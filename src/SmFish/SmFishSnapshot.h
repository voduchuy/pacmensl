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

#ifndef PACMENSL_SMFISHSNAPSHOT_H
#define PACMENSL_SMFISHSNAPSHOT_H

#include<map>
#include<armadillo>
#include"DiscreteDistribution.h"
#include"SensDiscreteDistribution.h"

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

PacmenslErrorCode SmFishSnapshotGradient(const SmFishSnapshot &data,
                                         const SensDiscreteDistribution &distribution,
                                         std::vector<PetscReal> &gradient,
                                         arma::Col<int> measured_species = arma::Col<int>({}),
                                         bool use_base_2 = false);
}

#endif //PACMENSL_SMFISHSNAPSHOT_H
