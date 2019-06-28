//
// Created by Huy Vo on 6/3/19.
//

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
  TcoefFun       t_fun_;
  void           *t_fun_args_;
  PropFun        prop_;
  void           *prop_args_;

  Model();

  explicit Model(arma::Mat<int> stoichiometry_matrix, TcoefFun t_fun, void *t_fun_args_, PropFun prop,
                 void *prop_args);

  Model(const Model &model);

  Model &operator=(const Model &model) noexcept;

  Model &operator=(Model &&model) noexcept;
};
};

#endif //PACMENSL_MODELS_H
