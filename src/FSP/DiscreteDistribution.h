//
// Created by Huy Vo on 6/4/19.
//

#ifndef PACMENSL_FSPSOLUTION_H
#define PACMENSL_FSPSOLUTION_H

#include<armadillo>
#include<petsc.h>
#include "cme_util.h"
#include "StateSetBase.h"

namespace pacmensl {
    struct DiscreteDistribution {
        MPI_Comm comm_ = nullptr;
        double t_ = 0.0;
        arma::Mat<int> states;
        Vec p = nullptr;

        DiscreteDistribution();

        DiscreteDistribution(MPI_Comm comm, double t, const StateSetBase *state_set, const Vec& p);

        DiscreteDistribution(const DiscreteDistribution &dist);

        DiscreteDistribution(DiscreteDistribution &&dist) noexcept;

        DiscreteDistribution &operator=(const DiscreteDistribution &);

        DiscreteDistribution &operator=(DiscreteDistribution &&) noexcept;

        ~DiscreteDistribution();
    };

    arma::Col<PetscReal> Compute1DMarginal(const DiscreteDistribution dist, int species);
}

#endif //PACMENSL_FSPSOLUTION_H
