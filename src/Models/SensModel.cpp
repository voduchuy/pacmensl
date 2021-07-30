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

pacmensl::SensModel::SensModel(const int num_parameters,
                               const arma::Mat<int> &stoichiometry_matrix,
                               const std::vector<int> &tv_reactions,
                               const pacmensl::TcoefFun &prop_t,
                               const pacmensl::PropFun &prop_x,
                               const pacmensl::DTcoefFun &dprop_t,
                               const std::vector<std::vector<int>> &dprop_t_sp,
                               const pacmensl::DPropFun &dprop_x,
                               const std::vector<std::vector<int>> &dprop_x_sp,
                               void *prop_t_args,
                               void *prop_x_args,
                               void *dprop_t_args,
                               void *dprop_x_args) {
  num_parameters_ = num_parameters;
  num_reactions_ = stoichiometry_matrix.n_cols;
  stoichiometry_matrix_ = stoichiometry_matrix;
  prop_t_ = prop_t;
  prop_x_ = prop_x;
  dprop_t_ = dprop_t;
  dprop_t_sp_ = dprop_t_sp;
  dprop_x_ = dprop_x;
  dprop_x_sp_ = dprop_x_sp;
  
  prop_t_args_ = prop_t_args;
  prop_x_args_ = prop_x_args;
  dprop_t_args_ = dprop_t_args;
  dprop_x_args_ = dprop_x_args;
  
  tv_reactions_ = tv_reactions;
  
}
