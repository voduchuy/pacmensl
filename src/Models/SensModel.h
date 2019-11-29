//
// Created by Huy Vo on 2019-06-27.
//

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
