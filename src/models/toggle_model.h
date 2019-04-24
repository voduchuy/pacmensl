#pragma once

#include <armadillo>
#include <petscmat.h>

namespace toggle_cme {
/* Stoichiometric matrix of the toggle switch model */
    arma::Mat<PetscInt> SM{{1, 1, -1, 0, 0, 0},
                           {0, 0, 0,  1, 1, -1}};

    const int nReaction = 6;

/* Parameters for the propensity functions */
    const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3.0e0}, nxy{2.1e0},
            kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{1.6e-2}, dy{3.8e-4};

    // Function to constraint the shape of the FSP
    void  lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states,
                     int *vals){

        for (int i{0}; i < num_states; ++i){
            vals[i*num_constrs] = states[num_species*i];
            vals[i*num_constrs + 1] = states[num_species*i+1];
            vals[i*num_constrs + 2] = states[num_species*i]*states[num_species*i+1];
        }
    }
    arma::Row<int> rhs_constr{200, 200, 2000};
    arma::Row<double> expansion_factors{0.2, 0.2, 0.2};

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
        return {kx0, kx, dx, ky0, ky, dy};
    }
}
