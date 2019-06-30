//
// Created by Huy Vo on 6/4/19.
//

#ifndef PACMENSL_FSPSOLUTION_H
#define PACMENSL_FSPSOLUTION_H

#include<armadillo>
#include<petsc.h>
#include "Sys.h"
#include "StateSetBase.h"

namespace pacmensl {
    struct DiscreteDistribution {
        MPI_Comm comm_ = nullptr;
        double t_ = 0.0;
        arma::Mat<int> states_;
        Vec p_ = nullptr;

        DiscreteDistribution();

        DiscreteDistribution(MPI_Comm comm, double t, const StateSetBase *state_set, const Vec& p);

        DiscreteDistribution(const DiscreteDistribution &dist);

        DiscreteDistribution(DiscreteDistribution &&dist) noexcept;

        DiscreteDistribution &operator=(const DiscreteDistribution &);

        DiscreteDistribution &operator=(DiscreteDistribution &&) noexcept;

        int GetStateView( int &num_states, int &num_species, int *&states);

        int GetProbView( int &num_states, double *&p);

        int RestoreProbView( double *&p);

        ~DiscreteDistribution();
    };

    arma::Col<PetscReal> Compute1DMarginal(const DiscreteDistribution &dist, int species);
}

#endif //PACMENSL_FSPSOLUTION_H
