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

#ifndef PACMENSL_FSPSOLUTION_H
#define PACMENSL_FSPSOLUTION_H

#include<armadillo>
#include<petsc.h>
#include "Sys.h"
#include "StateSetBase.h"

namespace pacmensl {

    struct DiscreteDistribution {
        MPI_Comm comm_ = MPI_COMM_NULL;
        double t_ = 0.0;
        arma::Mat<int> states_;
        Vec p_ = nullptr;

        DiscreteDistribution();

        DiscreteDistribution(MPI_Comm comm, double t, const StateSetBase *state_set, const Vec& p);

        DiscreteDistribution(const DiscreteDistribution &dist);

        DiscreteDistribution(DiscreteDistribution &&dist) noexcept;

        DiscreteDistribution &operator=(const DiscreteDistribution &);

        DiscreteDistribution &operator=(DiscreteDistribution &&) noexcept;

        PacmenslErrorCode GetStateView( int &num_states, int &num_species, int *&states);

        PacmenslErrorCode GetProbView( int &num_states, double *&p);

        PacmenslErrorCode RestoreProbView( double *&p);

        PacmenslErrorCode WeightedAverage(int nout, PetscReal *fout,
                                          std::function<PacmenslErrorCode(int num_species, int *x,
                                                                          int nout, PetscReal *wx,
                                                                          void *args)> weight_func,
                                          void *wf_args);



        ~DiscreteDistribution();
    };

    arma::Col<PetscReal> Compute1DMarginal(const DiscreteDistribution &dist, int species);
}

#endif //PACMENSL_FSPSOLUTION_H
