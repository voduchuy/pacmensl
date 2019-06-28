#pragma once

#include <armadillo>
#include <petscmat.h>

namespace repressilator_cme {
// stoichiometric matrix of the toggle switch model
    arma::Mat<PetscInt> SM{
            {1, -1, 0, 0,  0,  0},
            {0, 0,  1, -1, 0, 0},
            {0, 0,  0,  0, 1, -1},
    };

// reaction parameters
    const PetscReal k1{100.0}, ka{20.0}, ket{6.0}, kg{1.0};

    // Function to constraint the shape of the Fsp
    void  lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states,
                       int *vals){

        for (int i{0}; i < num_states; ++i){
            vals[i*num_constrs] = (states[num_species*i]);
            vals[i*num_constrs + 1] = (states[num_species*i+1]);
            vals[i*num_constrs + 2] = (states[num_species*i+2]);
            vals[i*num_constrs + 3] = (states[num_species*i])*(states[num_species*i+1]);
            vals[i*num_constrs + 4] = (states[num_species*i+2])*(states[num_species*i+1]);
            vals[i*num_constrs + 5] = (states[num_species*i])*(states[num_species*i+2]);
        }
    }
    arma::Row<int> rhs_constr{22, 2, 2, 44, 4, 44};
    arma::Row<double> expansion_factors{0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    arma::Row<int> rhs_constr_hyperrec{22, 2, 2};
    arma::Row<double> expansion_factors_hyperrec{0.2, 0.2, 0.2};

// propensity function
    PetscReal propensity(PetscInt *X, PetscInt k) {
        switch (k) {
            case 0:
                return k1/(1.0 + ka*pow(1.0*PetscReal(X[1]), ket));
            case 1:
                return kg*PetscReal(X[0]);
            case 2:
                return k1/(1.0 + ka*pow(1.0*PetscReal(X[2]), ket));
            case 3:
                return kg*PetscReal(X[1]);
            case 4:
                return k1/(1.0 + ka*pow(1.0*PetscReal(X[0]), ket));
            case 5:
                return kg*PetscReal(X[2]);
            default:
                return 0.0;
        }
    }

// function to compute the time-dependent coefficients of the propensity functions
    arma::Row<double> t_fun(double t) {
        arma::Row<double> u(6, arma::fill::ones);
        return u;
    }

}
