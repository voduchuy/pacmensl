//
// Created by Huy Vo on 6/3/19.
//

#include "Model.h"

pacmensl::Model::Model() {
    stoichiometry_matrix_.set_size(0);
    t_fun_args_ = nullptr;
    t_fun_ = nullptr;
    prop_args_ = nullptr;
    prop_ = nullptr;
}

pacmensl::Model::Model( arma::Mat< int > stoichiometry_matrix, TcoefFun t_fun, void *t_fun_args_, PropFun prop,
                        void *prop_args ) {
    Model::stoichiometry_matrix_ = std::move(stoichiometry_matrix);
    Model::t_fun_ = std::move(t_fun);
    Model::t_fun_args_ = t_fun_args_;
    Model::prop_ = std::move(prop);
    Model::prop_args_ = prop_args;
}

pacmensl::Model::Model(const pacmensl::Model &model) {
  stoichiometry_matrix_ = (model.stoichiometry_matrix_);
  t_fun_ = (model.t_fun_);
  t_fun_args_ = (model.t_fun_args_);
  prop_ = model.prop_;
  prop_args_ = model.prop_args_;
}
pacmensl::Model &pacmensl::Model::operator=(pacmensl::Model &&model) noexcept {
  Model::stoichiometry_matrix_ = std::move(model.stoichiometry_matrix_);
  Model::t_fun_ = std::move(model.t_fun_);
  Model::t_fun_args_ = std::move(model.t_fun_args_);
  Model::prop_ = std::move(model.prop_);
  Model::prop_args_ = std::move(model.prop_args_);
  return *this;
}
pacmensl::Model &pacmensl::Model::operator=(const Model &model) noexcept {
  stoichiometry_matrix_ = (model.stoichiometry_matrix_);
  t_fun_ = (model.t_fun_);
  t_fun_args_ = (model.t_fun_args_);
  prop_ = model.prop_;
  prop_args_ = model.prop_args_;
  return *this;
}
