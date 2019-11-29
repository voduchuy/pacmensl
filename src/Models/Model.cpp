//
// Created by Huy Vo on 6/3/19.
//

#include "Model.h"

pacmensl::Model::Model() {
    stoichiometry_matrix_.set_size(0);
    prop_t_args_ = nullptr;
    prop_t_ = nullptr;
    prop_x_args_ = nullptr;
    prop_x_ = nullptr;
}

pacmensl::Model::Model(arma::Mat<int> stoichiometry_matrix,
                       TcoefFun prop_t,
                       PropFun prop_x,
                       void *prop_t_args,
                       void *prop_x_args,
                       const std::vector<int> &tv_reactions_)
{
    Model::stoichiometry_matrix_ = std::move(stoichiometry_matrix);
    Model::prop_t_ = std::move(prop_t);
    Model::prop_t_args_ = prop_t_args;
    Model::prop_x_ = std::move(prop_x);
    Model::prop_x_args_ = prop_x_args;
    Model::tv_reactions_ = tv_reactions_;
}

pacmensl::Model::Model(const pacmensl::Model &model) {
  stoichiometry_matrix_ = (model.stoichiometry_matrix_);
  prop_t_ = (model.prop_t_);
  prop_t_args_ = (model.prop_t_args_);
  prop_x_ = model.prop_x_;
  prop_x_args_ = model.prop_x_args_;
  tv_reactions_ = model.tv_reactions_;
}
pacmensl::Model &pacmensl::Model::operator=(pacmensl::Model &&model) noexcept {
  if (this == &model) return *this;
  Model::stoichiometry_matrix_ = std::move(model.stoichiometry_matrix_);
  Model::prop_t_ = std::move(model.prop_t_);
  Model::prop_t_args_ = model.prop_t_args_;
  Model::prop_x_ = std::move(model.prop_x_);
  Model::prop_x_args_ = model.prop_x_args_;
  Model::tv_reactions_ = std::move(model.tv_reactions_);
  return *this;
}
pacmensl::Model &pacmensl::Model::operator=(const Model &model) noexcept {
  stoichiometry_matrix_ = (model.stoichiometry_matrix_);
  prop_t_ = (model.prop_t_);
  prop_t_args_ = (model.prop_t_args_);
  prop_x_ = model.prop_x_;
  prop_x_args_ = model.prop_x_args_;
  tv_reactions_ = model.tv_reactions_;
  return *this;
}
