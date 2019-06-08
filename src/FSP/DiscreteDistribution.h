//
// Created by Huy Vo on 6/4/19.
//

#ifndef PFSPAT_FSPSOLUTION_H
#define PFSPAT_FSPSOLUTION_H

#include<armadillo>
#include<petsc.h>
#include "cme_util.h"

namespace cme{
    namespace parallel{
        struct DiscreteDistribution {
            MPI_Comm comm;
            double t;
            arma::Mat<int> states;
            Vec p;

            DiscreteDistribution();
            DiscreteDistribution(const DiscreteDistribution& dist);
            DiscreteDistribution(DiscreteDistribution&& dist);
            DiscreteDistribution& operator =(const DiscreteDistribution&);
            DiscreteDistribution& operator = (DiscreteDistribution&&) noexcept;
            ~DiscreteDistribution();
        };

        arma::Col<PetscReal> Compute1DMarginal(const DiscreteDistribution dist, int species);
    }
}

#endif //PFSPAT_FSPSOLUTION_H
