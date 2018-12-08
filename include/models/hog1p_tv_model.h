#pragma once
#include <armadillo>
#include <petscmat.h>

namespace hog1p_cme {
// stoichiometric matrix of the toggle switch model
    arma::Mat<PetscInt> SM {
            {  1,  -1,  -1, 0, 0,  0,  0,  0,  0 },
            {  0,  0,   0,  1, 0, -1,  0,  0,  0 },
            {  0,  0,   0,  0, 1,  0, -1,  0,  0 },
            {  0,  0,   0,  0, 0,  1,  0, -1,  0 },
            {  0,  0,   0,  0, 0,  0,  1,  0, -1 },
    };

// reaction parameters
    const PetscReal k12 {1.29}, k21 {1.0e0}, k23 {0.0067},
            k32 {0.027}, k34 {0.133}, k43 {0.0381},
            kr2 {0.0116}, kr3 {0.987}, kr4 {0.0538},
            trans {0.01}, gamma {0.0049},
// parameters for the time-dependent factors
            r1 {6.9e-5}, r2 {7.1e-3}, eta {3.1}, Ahog {9.3e09}, Mhog {6.4e-4};

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
                return kr2 * double(X[0] == 1) + kr3 * double(X[0] == 2) + kr4 * double(X[0] == 3);
            case 5:
                return trans * double(X[1]);
            case 6:
                return trans * double(X[2]);
            case 7:
                return gamma * double(X[3]);
            case 8:
                return gamma * double(X[4]);
            default:
                return 0.0;
        }
    }

// function to compute the time-dependent coefficients of the propensity functions
    arma::Row<double> t_fun( double t)
    {
        arma::Row<double> u( 9, arma::fill::ones );

        double h1 = (1.0 - exp(-r1*t))*exp(-r2*t);

        double hog1p = pow( h1/( 1.0 + h1/Mhog), eta )*Ahog;

        u(2) = std::max( 0.0, 3200.0 - 7710.0*(hog1p) );
        //u(2) = std::max(0.0, 3200.0 - (hog1p));

        return u;
    }

}
