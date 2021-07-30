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

#ifndef PACMENSL_SRC_SENSFSP_SENSMODEL_H_
#define PACMENSL_SRC_SENSFSP_SENSMODEL_H_

#include <armadillo>
#include "Sys.h"
#include "Model.h"

/**
 * @file SensModel.h
 */

namespace pacmensl {

using DTcoefFun = std::function<int(
    const int parameter_idx,
    const double t,
    int num_coefs,
    double *outputs,
    void *args
)>;

using DPropFun = std::function<int(
    const int parameter_idx,
    const int reaction_idx,
    const int num_species,
    const int num_states,
    const int *states,
    double *outputs,
    void *args)>;

/*
 * Class for representing a stochastic chemical reaction network model with sensitivity information.
 */
class SensModel {
 public:
  int num_reactions_ = 0; ///< number of reactions
  int num_parameters_ = 0; ///< number of parameters
  arma::Mat<int> stoichiometry_matrix_; ///< stoichiometry matrix, reactions are ordered column-wise

  std::vector<int> tv_reactions_; ///< container for reaction indices with time-varying propensities?

  // propensities
  PropFun prop_x_; ///< function to evaluate state-dependent factors
  void *prop_x_args_; ///< pointer to extra arguments needed by \ref prop_x_
  TcoefFun prop_t_; ///< function to evaluate time-dependent factors
  void *prop_t_args_; ///< pointer to extra arguments needed by \ref prop_t_

  // derivatives of propensities
  DTcoefFun dprop_t_;
  void *dprop_t_args_;
  std::vector<std::vector<int>> dprop_t_sp_;

  DPropFun dprop_x_;
  void *dprop_x_args_;
  std::vector<std::vector<int>> dprop_x_sp_;

  SensModel() {};

  explicit SensModel(
      const int num_parameters,
      const arma::Mat<int> &stoichiometry_matrix,
      const std::vector<int> &tv_reactions,
      const TcoefFun &prop_t,
      const PropFun &prop_x,
      const DTcoefFun &dprop_t,
      const std::vector<std::vector<int>> &dprop_t_sp,
      const DPropFun &dprop_x,
      const std::vector<std::vector<int>> &dprop_x_sp = std::vector<std::vector<int>>(),
      void *prop_t_args = nullptr,
      void *prop_x_args = nullptr,
      void *dprop_t_args = nullptr,
      void *dprop_x_args = nullptr);
};
}

#endif //PACMENSL_SRC_SENSFSP_SENSMODEL_H_
