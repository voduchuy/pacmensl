#pragma once
#include <armadillo>
#include <petscmat.h>

namespace hog1p_cme {
// stoichiometric matrix of the toggle switch model
const arma::Mat<int> SM {
        {  1,  -1,  -1, 0, 0,  0,  0,  0,  0 },
        {  0,  0,   0,  1, 0, -1,  0,  0,  0 },
        {  0,  0,   0,  0, 1,  0, -1,  0,  0 },
        {  0,  0,   0,  0, 0,  1,  0, -1,  0 },
        {  0,  0,   0,  0, 0,  0,  1,  0, -1 },
};

// reaction parameters
const double k12 {1.29}, k21 {1.0e0}, k23 {0.0067},
k32 {0.027}, k34 {0.133}, k43 {0.0381},
kr2 {0.0116}, kr3 {0.987}, kr4 {0.0538},
trans {0.01}, gamma {0.0049},
// parameters for the time-dependent factors
r1 {6.9e-5}, r2 {7.1e-3}, eta {3.1}, Ahog {9.3e09}, Mhog {6.4e-4};

// propensity function for toggle
PetscReal propensity( PetscInt *X, PetscInt k )
{
        switch ( X[0] )
        {
        case 0:
        {
          switch (k)
          {
            case 0: return k12;
            case 1: return 0.0;
            case 2: return 0.0;
            case 3: return 0.0;
            case 4: return 0.0;
          }
        }
        case 1:
        {
          switch (k)
          {
            case 0: return k23;
            case 1: return 0.0;
            case 2: return k21;
            case 3: return kr2;
            case 4: return kr2;
          }
        }
        case 2:
        {
          switch (k)
          {
            case 0: return k34;
            case 1: return k32;
            case 2: return 0.0;
            case 3: return kr3;
            case 4: return kr3;
          }
        }
        case 3:
        {
          switch (k)
          {
            case 0: return 0.0;
            case 1: return k43;
            case 2: return 0.0;
            case 3: return kr4;
            case 4: return kr4;
          }
        }
        }

        switch (k)
        {
          case 5: return trans*PetscReal( X[1] );
          case 6: return trans*PetscReal( X[2] );
          case 7: return gamma*PetscReal( X[3] );
          case 8: return gamma*PetscReal( X[4] );
        }
}

// function to compute the time-dependent coefficients of the propensity functions
arma::Row<double> t_fun( double t)
{
        arma::Row<double> u( 9, arma::fill::ones );

        double h1 = (1.0 - exp(-r1*t))*exp(-r2*t);

        double hog1p = pow( h1/( 1.0 + h1/Mhog), eta )*Ahog;

        u(2) = std::max( 0.0, 3200.0 - 7710.0*(hog1p) );

        return u;
}

}
