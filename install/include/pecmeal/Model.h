//
// Created by Huy Vo on 6/3/19.
//

#ifndef PECMEAL_MODELS_H
#define PECMEAL_MODELS_H

#include <armadillo>

namespace pecmeal {
    using PropFun = std::function<double(const int *, const int)>;
    using TcoefFun = std::function<arma::Row<double>(double t)>;

    class Model {
    public:
        arma::Mat<int> stoichiometry_matrix_;
        TcoefFun t_fun_;
        PropFun prop_;

        Model();

        explicit Model(arma::Mat<int> stoichiometry_matrix, TcoefFun t_fun, PropFun prop);

        Model(const Model &model);
    };
};


#endif //PECMEAL_MODELS_H
