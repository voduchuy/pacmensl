#pragma once
#include <armadillo>
#include <petscmat.h>

namespace toggle_cme{
/* Stoichiometric matrix of the toggle switch model */
    arma::Mat<int> SM{{1, 1, -1, 0, 0, 0},
                      {0, 0, 0,  1, 1, -1}};

    const int nReaction = 6;

/* Parameters for the propensity functions */
    const double ayx{6.1e-3}, axy{2.6e-3}, nyx{4.1e0}, nxy{3.0e0},
            kx0{6.8e-3}, kx{1.6}, dx{0.00067}, ky0{2.2e-3}, ky{1.7}, dy{3.8e-4};

// propensity function for toggle
    PetscReal propensity(PetscInt *X, PetscInt k) {
        switch (k) {
            case 0:
                return 1.0;
            case 1:
                return 1.0 / (1.0 + ayx * pow(PetscReal(X[1]), nyx));
            case 2:
                return PetscReal(X[0]);
            case 3:
                return 1.0;
            case 4:
                return 1.0 / (1.0 + axy * pow(PetscReal(X[0]), nxy));
            case 5:
                return PetscReal(X[1]);
        }
        return 0.0;
    }

    arma::Row<PetscReal> t_fun(PetscReal t) {
        //return {kx0, kx, dx, ky0, ky, dy};
        return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
    }
}
