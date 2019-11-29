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
