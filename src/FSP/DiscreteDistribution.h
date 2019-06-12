//
// Created by Huy Vo on 6/4/19.
//

#ifndef PECMEAL_FSPSOLUTION_H
#define PECMEAL_FSPSOLUTION_H

#include<armadillo>
#include<petsc.h>
#include "cme_util.h"

namespace pecmeal {
    struct DiscreteDistribution {
        MPI_Comm comm;
        double t;
        arma::Mat<int> states;
        Vec p;

        DiscreteDistribution();

        DiscreteDistribution(const DiscreteDistribution &dist);

        DiscreteDistribution(DiscreteDistribution &&dist) noexcept;

        DiscreteDistribution &operator=(const DiscreteDistribution &);

        DiscreteDistribution &operator=(DiscreteDistribution &&) noexcept;

        ~DiscreteDistribution();
    };

    arma::Col<PetscReal> Compute1DMarginal(const DiscreteDistribution dist, int species);
}

#endif //PECMEAL_FSPSOLUTION_H
