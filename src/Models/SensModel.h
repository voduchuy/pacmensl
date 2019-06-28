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
class SensModel {
 public:
  int            num_reactions_  = 0;
  int            num_parameters_ = 0;
  arma::Mat<int> stoichiometry_matrix_;
  // propensities
  PropFun        propensity_xfac_;
  void *propensity_xfac_args_;
  TcoefFun propensity_tfac_;
  void *propensity_tfac_args_;
  // derivatives of propensities
  std::vector<PropFun>  dpropensity_xfac_;
  std::vector<TcoefFun> dpropensity_tfac_;
  std::vector<void *>   dpropensity_xfac_args_;
  std::vector<void *>   dpropensity_tfac_args_;
  arma::Mat<char>       dpropensity_nz_;

  SensModel() {};

  explicit SensModel(const arma::Mat<int> &stoichiometry_matrix,
                     const TcoefFun &t_fun,
                     void *t_fun_args,
                     const PropFun &prop,
                     void *prop_args,
                     const std::vector<PropFun> &dpropensity_xfac,
                     const std::vector<void *> &dpropensity_xfac_args,
                     const std::vector<TcoefFun> &dpropensity_tfac,
                     const std::vector<void *> &dpropensity_tfac_args,
                     const arma::Mat<char> &dpropensity_nz);

  SensModel(const SensModel &model);

  SensModel &operator=(const SensModel &model) noexcept;

  SensModel &operator=(SensModel &&model) noexcept;
};
}

#endif //PACMENSL_SRC_SENSFSP_SENSMODEL_H_
