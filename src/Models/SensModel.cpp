//
// Created by Huy Vo on 2019-06-27.
//

#include "SensModel.h"

pacmensl::SensModel::SensModel(const arma::Mat<int> &stoichiometry_matrix,
                               const pacmensl::TcoefFun &t_fun,
                               void *t_fun_args,
                               const pacmensl::PropFun &prop,
                               void *prop_args,
                               const std::vector<pacmensl::PropFun> &dpropensity_xfac,
                               const std::vector<void *> &dpropensity_xfac_args,
                               const std::vector<pacmensl::TcoefFun> &dpropensity_tfac,
                               const std::vector<void *> &dpropensity_tfac_args,
                               const arma::Mat<char> &dpropensity_nz) {
  num_reactions_ = stoichiometry_matrix.n_cols;
  num_parameters_ = dpropensity_xfac.size();
  stoichiometry_matrix_ = stoichiometry_matrix;
  propensity_tfac_ = t_fun;
  propensity_xfac_ = prop;
  propensity_tfac_args_ = t_fun_args;
  propensity_xfac_args_ = prop_args;
  dpropensity_tfac_ = dpropensity_tfac;
  dpropensity_xfac_ = dpropensity_xfac;
  dpropensity_tfac_args_ = dpropensity_tfac_args;
  dpropensity_xfac_args_ = dpropensity_xfac_args;
  dpropensity_nz_ = dpropensity_nz;
}

pacmensl::SensModel::SensModel(const pacmensl::SensModel &model) {
  num_reactions_ = model.stoichiometry_matrix_.n_cols;
  num_parameters_ = model.dpropensity_xfac_.size();
  stoichiometry_matrix_ = model.stoichiometry_matrix_;
  propensity_tfac_ = model.propensity_tfac_;
  propensity_xfac_ = model.propensity_xfac_;
  propensity_tfac_args_ = model.propensity_tfac_args_;
  propensity_xfac_args_ = model.propensity_xfac_args_;
  dpropensity_tfac_ = model.dpropensity_tfac_;
  dpropensity_xfac_ = model.dpropensity_xfac_;
  dpropensity_tfac_args_ = model.dpropensity_tfac_args_;
  dpropensity_xfac_args_ = model.dpropensity_xfac_args_;
  dpropensity_nz_ = model.dpropensity_nz_;
}

pacmensl::SensModel &pacmensl::SensModel::operator=(const pacmensl::SensModel &model) noexcept {
  stoichiometry_matrix_.clear();
  dpropensity_nz_.clear();
  dpropensity_xfac_args_.clear();
  dpropensity_xfac_.clear();
  dpropensity_tfac_.clear();
  dpropensity_tfac_args_.clear();

  num_reactions_ = model.stoichiometry_matrix_.n_cols;
  num_parameters_ = model.dpropensity_xfac_.size();
  stoichiometry_matrix_ = model.stoichiometry_matrix_;
  propensity_tfac_ = model.propensity_tfac_;
  propensity_xfac_ = model.propensity_xfac_;
  propensity_tfac_args_ = model.propensity_tfac_args_;
  propensity_xfac_args_ = model.propensity_xfac_args_;
  dpropensity_tfac_ = model.dpropensity_tfac_;
  dpropensity_xfac_ = model.dpropensity_xfac_;
  dpropensity_tfac_args_ = model.dpropensity_tfac_args_;
  dpropensity_xfac_args_ = model.dpropensity_xfac_args_;
  dpropensity_nz_ = model.dpropensity_nz_;
  return *this;
}

pacmensl::SensModel &pacmensl::SensModel::operator=(pacmensl::SensModel &&model) noexcept {
  stoichiometry_matrix_.clear();
  dpropensity_nz_.clear();
  dpropensity_xfac_args_.clear();
  dpropensity_xfac_.clear();
  dpropensity_tfac_.clear();
  dpropensity_tfac_args_.clear();

  num_reactions_ = std::move(model.stoichiometry_matrix_.n_cols);
  num_parameters_ = std::move(model.dpropensity_xfac_.size());
  stoichiometry_matrix_ = std::move(model.stoichiometry_matrix_);
  propensity_tfac_ = std::move(model.propensity_tfac_);
  propensity_xfac_ = std::move(model.propensity_xfac_);
  propensity_tfac_args_ = std::move(model.propensity_tfac_args_);
  propensity_xfac_args_ = std::move(model.propensity_xfac_args_);
  dpropensity_tfac_ = std::move(model.dpropensity_tfac_);
  dpropensity_xfac_ = std::move(model.dpropensity_xfac_);
  dpropensity_tfac_args_ = std::move(model.dpropensity_tfac_args_);
  dpropensity_xfac_args_ = std::move(model.dpropensity_xfac_args_);
  dpropensity_nz_ = std::move(model.dpropensity_nz_);

  return *this;
}
