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
