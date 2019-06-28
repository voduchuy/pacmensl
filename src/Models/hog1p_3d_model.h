#pragma once

#include <armadillo>
#include <petscmat.h>

namespace hog3d_cme {
// stoichiometric matrix of the toggle switch model
    arma::Mat<PetscInt> SM{
            {1, -1, -1, 0, 0,  0,  0},
            {0, 0,  0,  1, -1, -1, 0},
            {0, 0,  0,  0, 1,  0,  -1},
    };

// reaction parameters
    const PetscReal k12{1.29}, k21{1.0e0}, k23{0.0067},
            k32{0.027}, k34{0.133}, k43{0.0381},
            kr2{0.0116}, kr3{0.987}, kr4{0.0538},
            trans{0.01}, gamma{0.0049},
// parameters for the time-dependent factors
            r1{6.9e-5}, r2{7.1e-3}, eta{3.1}, Ahog{9.3e09}, Mhog{6.4e-4};

    // Function to constraint the shape of the Fsp
    void  lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states,
                       int *vals, void *args){

        for (int i{0}; i < num_states; ++i){
            vals[i*num_constrs] = states[num_species*i];
            vals[i*num_constrs + 1] = states[num_species*i+1];
            vals[i*num_constrs + 2] = states[num_species*i+2];
            vals[i*num_constrs + 3] = (states[num_species*i]==0)*((states[num_species*i+1]) + (states[num_species*i+2]));
            vals[i*num_constrs + 4] = (states[num_species*i]==1)*((states[num_species*i+1]) + (states[num_species*i+2]));
            vals[i*num_constrs + 5] = (states[num_species*i]==2)*((states[num_species*i+1]) + (states[num_species*i+2]));
            vals[i*num_constrs + 6] = (states[num_species*i]==3)*((states[num_species*i+1]) + (states[num_species*i+2]));;
        }
    }
    arma::Row<int> rhs_constr{3, 4, 4, 1, 10, 10, 10};
    arma::Row<double> expansion_factors{0.0, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5};

// propensity function
    PetscReal propensity(PetscInt *X, PetscInt k) {
        switch (k) {
            case 0:
                return k12 * double(X[0] == 0) + k23 * double(X[0] == 1) + k34 * double(X[0] == 2);
            case 1:
                return k32 * double(X[0] == 2) + k43 * double(X[0] == 3);
            case 2:
                return k21 * double(X[0] == 1);
            case 3:
                return kr2 * double(X[0] == 1) + kr3 * double(X[0] == 2) + kr4 * double(X[0] == 3);
            case 4:
                return trans * double(X[1]);
            case 5:
                return gamma * double(X[1]);
            case 6:
                return gamma * double(X[2]);
            default:
                return 0.0;
        }
    }

// function to compute the time-dependent coefficients of the propensity functions
    arma::Row<double> t_fun(double t) {
        arma::Row<double> u(7, arma::fill::ones);

        double h1 = (1.0 - exp(-r1 * t)) * exp(-r2 * t);

        double hog1p = pow(h1 / (1.0 + h1 / Mhog), eta) * Ahog;

        u(2) = std::max(0.0, 3200.0 - 7710.0 * (hog1p));
        //u(2) = std::max(0.0, 3200.0 - (hog1p));

        return u;
    }

}
