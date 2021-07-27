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

#ifndef PACMENSL_MODELS_H
#define PACMENSL_MODELS_H

#include <armadillo>

namespace pacmensl {
using PropFun = std::function<int(const int reaction,
                                  const int num_species,
                                  const int num_states,
                                  const int *states,
                                  double *outputs,
                                  void *args)>;
using TcoefFun = std::function<int(double t, int num_coefs, double *outputs, void *args)>;

class Model {
 public:
  arma::Mat<int> stoichiometry_matrix_;
  TcoefFun       prop_t_;
  void           *prop_t_args_;
  PropFun        prop_x_;
  void           *prop_x_args_;
  std::vector<int> tv_reactions_;

  Model();

  explicit Model(arma::Mat<int> stoichiometry_matrix,
                 TcoefFun prop_t,
                 PropFun prop_x,
                 void *prop_t_args = nullptr,
                 void *prop_x_args = nullptr,
                 const std::vector<int> &tv_reactions_ = std::vector<int>());

  Model(const Model &model);

  Model &operator=(const Model &model) noexcept;

  Model &operator=(Model &&model) noexcept;
};
};

#endif //PACMENSL_MODELS_H
