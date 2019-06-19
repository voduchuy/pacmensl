//
// Created by Huy Vo on 6/3/19.
//

#ifndef PACMENSL_MODELS_H
#define PACMENSL_MODELS_H

#include <armadillo>

namespace pacmensl {
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


#endif //PACMENSL_MODELS_H
