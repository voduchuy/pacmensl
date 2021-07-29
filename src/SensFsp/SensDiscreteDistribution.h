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

#ifndef PACMENSL_SRC_SENSFSP_SENSDISCRETEDISTRIBUTION_H_
#define PACMENSL_SRC_SENSFSP_SENSDISCRETEDISTRIBUTION_H_

#include "DiscreteDistribution.h"

namespace pacmensl {
class SensDiscreteDistribution : public DiscreteDistribution {
 public:

std::vector<Vec> dp_;

SensDiscreteDistribution();

SensDiscreteDistribution(MPI_Comm comm, double t, const StateSetBase *state_set, const Vec &p,
                       const std::vector<Vec> &dp);

SensDiscreteDistribution(const SensDiscreteDistribution &dist);

SensDiscreteDistribution(SensDiscreteDistribution &&dist) noexcept;

SensDiscreteDistribution &operator=(const SensDiscreteDistribution &dist);

SensDiscreteDistribution &operator=(SensDiscreteDistribution &&dist) noexcept;

PacmenslErrorCode GetSensView(int is, int &num_states, double *&p);
PacmenslErrorCode RestoreSensView(int is, double *&p);

PacmenslErrorCode WeightedAverage(int is, int nout, PetscReal *fout,
                                  std::function<PacmenslErrorCode(int num_species, int *x,
                                                                  int nout, PetscReal *wx,
                                                                  void *args)> weight_func,
                                  void *wf_args);

  ~SensDiscreteDistribution();
};

PacmenslErrorCode Compute1DSensMarginal(const pacmensl::SensDiscreteDistribution &dist,
                                        int is,
                                        int species,
                                        arma::Col<PetscReal> &out);
PacmenslErrorCode ComputeFIM(SensDiscreteDistribution &dist, arma::Mat<PetscReal> &fim);
}

#endif //PACMENSL_SRC_SENSFSP_SENSDISCRETEDISTRIBUTION_H_
