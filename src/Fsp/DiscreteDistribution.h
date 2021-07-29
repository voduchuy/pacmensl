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

    /**
     * @brief Distributed object to store the solution of the CME.
     */
    struct DiscreteDistribution {
        MPI_Comm comm_ = MPI_COMM_NULL; ///< Communicator context of owning processes
        double t_ = 0.0; ///< Time stamp of the distribution
        arma::Mat<int> states_; ///< Set of states
        Vec p_ = nullptr; ///< PETSc vector object for the probabilities

        DiscreteDistribution();

        /**
         * @brief Constructor.
         * @param comm (in) communicator context for the owning processes.
         * @param t (in) time stamp.
         * @param state_set (in) state space.
         * @param p (in) probability vector associated with the state space.
         */
        DiscreteDistribution(MPI_Comm comm, double t, const StateSetBase *state_set, const Vec& p);

        DiscreteDistribution(const DiscreteDistribution &dist);

        DiscreteDistribution(DiscreteDistribution &&dist) noexcept;

        DiscreteDistribution &operator=(const DiscreteDistribution &);

        DiscreteDistribution &operator=(DiscreteDistribution &&) noexcept;

        /**
         * @brief Get a pointer to the state space of the distribution.
         * @param num_states (out) number of states.
         * @param num_species (out)
         * @param states (out) pointer to the array of states.
         * @attention only the states owned by the calling process are given upon return.
         * @return Error code: 0 (success), -1 (failure).
         */
        PacmenslErrorCode GetStateView( int &num_states, int &num_species, int *&states);

        /**
         * @brief Get a view of the probabilities for copying or modifying.
         * @details This method is __collective__.
         * @attention Every call of this method must be matched with a call to \ref RestoreProbView after all the desired operations are done.\n
         * @param num_states (out) number of states.
         * @param p (out) pointer to the array of probabilities.
         * @return Error code: 0 (success), -1 (failure).
         */
        PacmenslErrorCode GetProbView( int &num_states, double *&p);

        /**
         * @brief Restore the memory view of the probability vector.
         * @param p (in/out) pointer to the array of probabilities, will be set to nullptr upon return.
         * @return Error code: 0 (success), -1 (failure).
         */
        PacmenslErrorCode RestoreProbView( double *&p);

        /**
         * @brief Compute the average of a function across the state space.
         * @param nout (in) number of outputs.
         * @param fout (in/out) pointer to the array of outputs.
         * @param weight_func (in) function to average.
         * @param wf_args (in) pointer to extra arguments if needed.
         * @return
         */
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
