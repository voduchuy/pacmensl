//
// Created by Huy Vo on 6/3/19.
//

#include "Model.h"

pecmeal::Model::Model() {
    stoichiometry_matrix_.set_size(0);
    t_fun_ = nullptr;
    prop_ = nullptr;
}

pecmeal::Model::Model(arma::Mat<int> stoichiometry_matrix, TcoefFun t_fun,
                            PropFun prop) {
    Model::stoichiometry_matrix_ = std::move(stoichiometry_matrix);
    Model::t_fun_ = std::move(t_fun);
    Model::prop_ = std::move(prop);
}

pecmeal::Model::Model(const pecmeal::Model &model) {
    Model model1(model.stoichiometry_matrix_, TcoefFun (model.t_fun_), PropFun (model.prop_));
}
