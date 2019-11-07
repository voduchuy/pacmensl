//
// Created by Huy Vo on 2019-06-28.
//

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
