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

#include "SensModel.h"

pacmensl::SensModel::SensModel(const arma::Mat<int> &stoichiometry_matrix,
                               const std::vector<int> &tv_reactions,
                               const TcoefFun &prop_t,
                               const PropFun &prop_x,
                               const std::vector<TcoefFun> &dprop_t,
                               const std::vector<PropFun> &dprop_x,
                               const std::vector<int> &dprop_ic,
                               const std::vector<int> &dprop_rowptr,
                               void *prop_t_args,
                               void *prop_x_args,
                               const std::vector<void *> &dprop_t_args,
                               const std::vector<void *> &dprop_x_args)
{
  num_reactions_        = stoichiometry_matrix.n_cols;
  num_parameters_       = dprop_x.size();
  stoichiometry_matrix_ = stoichiometry_matrix;
  prop_t_               = prop_t;
  prop_x_               = prop_x;
  dprop_t_              = dprop_t;
  dprop_x_              = dprop_x;
  dpropensity_ic_       = dprop_ic;
  dpropensity_rowptr_   = dprop_rowptr;
  if (dpropensity_rowptr_.size() == num_parameters_) dpropensity_rowptr_.push_back(dpropensity_ic_.size());

  prop_t_args_ = prop_t_args;
  prop_x_args_ = prop_x_args;
  if (dprop_t_args.empty()) {
    dprop_t_args_ = std::vector<void *>(num_parameters_, nullptr);
  } else {
    dprop_t_args_ = dprop_t_args;
  }
  if (dprop_x_args.empty()) {
    dprop_x_args_ = std::vector<void *>(num_parameters_, nullptr);
  } else {
    dprop_x_args_ = dprop_x_args;
  }

  tv_reactions_ = tv_reactions;
}

pacmensl::SensModel::SensModel(const pacmensl::SensModel &model) {
  num_reactions_        = model.stoichiometry_matrix_.n_cols;
  num_parameters_       = model.dprop_x_.size();
  stoichiometry_matrix_ = model.stoichiometry_matrix_;
  prop_t_               = model.prop_t_;
  prop_x_               = model.prop_x_;
  prop_t_args_          = model.prop_t_args_;
  prop_x_args_          = model.prop_x_args_;
  dprop_t_              = model.dprop_t_;
  dprop_x_              = model.dprop_x_;
  dprop_t_args_         = model.dprop_t_args_;
  dprop_x_args_         = model.dprop_x_args_;
  dpropensity_ic_       = model.dpropensity_ic_;
  dpropensity_rowptr_   = model.dpropensity_rowptr_;
  tv_reactions_ = model.tv_reactions_;
}

pacmensl::SensModel &pacmensl::SensModel::operator=(const pacmensl::SensModel &model) noexcept {
  if (this == &model) return *this;

  stoichiometry_matrix_.clear();
  dprop_t_.clear();
  dprop_x_.clear();
  dpropensity_ic_.clear();
  dpropensity_rowptr_.clear();
  dprop_t_args_.clear();
  dprop_x_args_.clear();

  num_reactions_        = model.stoichiometry_matrix_.n_cols;
  num_parameters_       = model.dprop_x_.size();
  stoichiometry_matrix_ = model.stoichiometry_matrix_;
  prop_t_               = model.prop_t_;
  prop_x_               = model.prop_x_;
  prop_t_args_          = model.prop_t_args_;
  prop_x_args_          = model.prop_x_args_;
  dprop_t_              = model.dprop_t_;
  dprop_x_              = model.dprop_x_;
  dprop_t_args_         = model.dprop_t_args_;
  dprop_x_args_         = model.dprop_x_args_;
  dpropensity_ic_       = model.dpropensity_ic_;
  dpropensity_rowptr_   = model.dpropensity_rowptr_;

  tv_reactions_ = model.tv_reactions_;
  return *this;
}

pacmensl::SensModel &pacmensl::SensModel::operator=(pacmensl::SensModel &&model) noexcept {
  if (this == &model) return *this;

  stoichiometry_matrix_.clear();
  dprop_x_args_.clear();
  dprop_x_.clear();
  dprop_t_.clear();
  dprop_t_args_.clear();
  dpropensity_ic_.clear();
  dpropensity_rowptr_.clear();


  num_reactions_        = std::move(model.stoichiometry_matrix_.n_cols);
  num_parameters_       = std::move(model.dprop_x_.size());
  stoichiometry_matrix_ = std::move(model.stoichiometry_matrix_);
  prop_t_               = std::move(model.prop_t_);
  prop_x_               = std::move(model.prop_x_);
  prop_t_args_          = std::move(model.prop_t_args_);
  prop_x_args_          = std::move(model.prop_x_args_);
  dprop_t_              = std::move(model.dprop_t_);
  dprop_x_              = std::move(model.dprop_x_);
  dprop_t_args_         = std::move(model.dprop_t_args_);
  dprop_x_args_         = std::move(model.dprop_x_args_);
  dpropensity_ic_       = model.dpropensity_ic_;
  dpropensity_rowptr_   = model.dpropensity_rowptr_;
  tv_reactions_ = std::move(model.tv_reactions_);

  return *this;
}
