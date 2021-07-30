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
/**
 * @file Model.h
 */

#ifndef PACMENSL_MODELS_H
#define PACMENSL_MODELS_H

#include <armadillo>

namespace pacmensl {

/**
 * @brief Prototype for function to evaluate the state-dependent factors of propensities.
 * @param reaction (in) reaction index
 * @param num_species (in) number of species
 * @param num_states (in) number of states in the input array
 * @param states (in) pointer to the array of states
 * @param outputs (out) pointer to the output array. Propensity values are written here.
 * @param args (in) pointer to extra data if needed.
 */
using PropFun = std::function<int(const int reaction,
                                  const int num_species,
                                  const int num_states,
                                  const int *states,
                                  double *outputs,
                                  void *args)>;
/**
 * @brief Prototype for function to evaluate the time-dependent factors of propensities.
 * @param t (in) time
 * @param num_coefs (in) number of time-varying coefficients
 * @param outputs (out) pointer to the output array.
 * @param args (in) pointer to extra data if needed.
 */
using TcoefFun = std::function<int(double t, int num_coefs, double *outputs, void *args)>;

/**
 * @brief Object to store information about a stochastic reaction network.
 * @details We assume that the propensity functions are factorizable in the form \f$a_r(t,x;\theta) = c_r(t,\theta)d_r(x)\f$.
 */
class Model {
 public:
  arma::Mat<int>
      stoichiometry_matrix_; ///< Stoichiometry matrix of the reaction network, each column corresponds to a reaction
  TcoefFun prop_t_; ///< Callable to evaluate the time-dependent coefficients of the propensity functions. \n
  void *prop_t_args_; ///< Pointer to extra data (if there is any) needed for the excution of \ref prop_t_
  PropFun prop_x_; ///< Function to evaluate the state-dependent coefficients of the propensity functions.
  void *prop_x_args_; ///< Pointer to extra data (if there is any) needed for the excution of \ref prop_x_
  std::vector<int> tv_reactions_; ///< List of reactions whose propensities are time-varying

  /**
   * @brief Default constructor.
   */
  Model();

  /**
   * @brief Construct \ref Model object from a list of arguments.
   * @param stoichiometry_matrix (in) Stoichiometry matrix. Each column corresponds to a reaction.
   * @param prop_t (in) Function pointer to evaluate the time-varying coefficients. See document of \ref prop_t_.
   * @param prop_x (in) Function pointer to evaluate the time-independent factors of propensity functions. See document of \ref prop_x_.
   * @param prop_t_args (in) Pointer to extra argument needed for \ref prop_t
   * @param prop_x_args (in) Pointer to extra argument needed for \ref prop_x
   * @param tv_reactions_ (in) List of time-varying reactions
   */
  explicit Model(arma::Mat<int> stoichiometry_matrix,
                 TcoefFun prop_t,
                 PropFun prop_x,
                 void *prop_t_args = nullptr,
                 void *prop_x_args = nullptr,
                 const std::vector<int> &tv_reactions_ = std::vector<int>());

  Model(const Model &model);

  Model &operator=(const Model &model) noexcept;

  Model &operator=(Model &&model) noexcept;
};
};

#endif //PACMENSL_MODELS_H
