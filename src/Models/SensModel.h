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

namespace pacmensl {

/*
 * Class for representing a stochastic chemical reaction network model with sensitivity information.
 */
class SensModel
{
 public:
  int                   num_reactions_  = 0;
  int                   num_parameters_ = 0;
  arma::Mat<int>        stoichiometry_matrix_;
  // Which reactions have time-varying propensities?
  std::vector<int> tv_reactions_;
  // propensities
  PropFun               prop_x_;
  void                  *prop_x_args_;
  TcoefFun              prop_t_;
  void                  *prop_t_args_;
  // derivatives of propensities
  std::vector<PropFun>  dprop_x_;
  std::vector<TcoefFun> dprop_t_;
  std::vector<void *>   dprop_x_args_;
  std::vector<void *>   dprop_t_args_;
  std::vector<int>      dpropensity_ic_;
  std::vector<int>      dpropensity_rowptr_;

  SensModel() {};

  explicit SensModel(const arma::Mat<int> &stoichiometry_matrix,
                     const std::vector<int> &tv_reactions,
                     const TcoefFun &prop_t,
                     const PropFun &prop_x,
                     const std::vector<TcoefFun> &dprop_t,
                     const std::vector<PropFun> &dprop_x,
                     const std::vector<int> &dprop_ic = std::vector<int>(),
                     const std::vector<int> &dprop_rowptr = std::vector<int>(),
                     void *prop_t_args = nullptr,
                     void *prop_x_args = nullptr,
                     const std::vector<void *> &dprop_t_args = std::vector<void *>(),
                     const std::vector<void *> &dprop_x_args = std::vector<void *>());

  SensModel(const SensModel &model);

  SensModel &operator=(const SensModel &model) noexcept;

  SensModel &operator=(SensModel &&model) noexcept;
};
}

#endif //PACMENSL_SRC_SENSFSP_SENSMODEL_H_
