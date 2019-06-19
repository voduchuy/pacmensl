#pragma once

#include <armadillo>
#include <petscmat.h>

namespace hog1p_cme {
// stoichiometric matrix of the toggle switch model
    arma::Mat< PetscInt > SM{
            {1, -1, -1, 0, 0, 0,  0,  0,  0},
            {0, 0,  0,  1, 0, -1, 0,  0,  0},
            {0, 0,  0,  0, 1, 0,  -1, 0,  0},
            {0, 0,  0,  0, 0, 1,  0,  -1, 0},
            {0, 0,  0,  0, 0, 0,  1,  0,  -1},
    };

// reaction parameters
    const PetscReal k12{1.29}, k23{0.0067}, k34{0.133},
            k32{0.027}, k43{0.0381}, k21{1.0e0},
            kr21{0.005}, kr31{0.45}, kr41{0.025},
            kr22{0.0116}, kr32{0.987}, kr42{0.0538},
            trans{0.01},
            gamma1{0.001},
            gamma2{0.0049},
// parameters for the time-dependent factors
            r1{6.9e-5}, r2{7.1e-3}, eta{3.1}, Ahog{9.3e09}, Mhog{6.4e-4};

// propensity function
    PetscReal propensity( const PetscInt *X, const PetscInt k ) {
        switch ( k ) {
            case 0:
                return k12 * double( X[ 0 ] == 0 ) + k23 * double( X[ 0 ] == 1 ) + k34 * double( X[ 0 ] == 2 );
            case 1:
                return k32 * double( X[ 0 ] == 2 ) + k43 * double( X[ 0 ] == 3 );
            case 2:
                return k21 * double( X[ 0 ] == 1 );
            case 3:
                return kr21 * double( X[ 0 ] == 1 ) + kr31 * double( X[ 0 ] == 2 ) + kr41 * double( X[ 0 ] == 3 );
            case 4:
                return kr22 * double( X[ 0 ] == 1 ) + kr32 * double( X[ 0 ] == 2 ) + kr42 * double( X[ 0 ] == 3 );
            case 5:
                return trans * double( X[ 1 ] );
            case 6:
                return trans * double( X[ 2 ] );
            case 7:
                return gamma1 * double( X[ 3 ] );
            case 8:
                return gamma2 * double( X[ 4 ] );
            default:
                return 0.0;
        }
    }

// function to compute the time-dependent coefficients of the propensity functions
    arma::Row< double > t_fun( double t ) {
        arma::Row< double > u( 9, arma::fill::ones );

        double h1 = ( 1.0 - exp( -r1 * t )) * exp( -r2 * t );

        double hog1p = pow( h1 / ( 1.0 + h1 / Mhog ), eta ) * Ahog;

        u( 2 ) = std::max( 0.0, 3200.0 - 7710.0 * ( hog1p ));

        return u;
    }

    // Function to constraint the shape of the FSP
    void  lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states,
                     int *vals, void *args){

        for (int j{0}; j < num_states; ++j){
            for (int i{0}; i < 5; ++i){
                vals[num_constrs*j + i] = (states[num_species*j+i]);
            }
            vals[num_constrs*j + 5] = ((states[num_species*j+1]) + (states[num_species*j+3]));
            vals[num_constrs*j + 6] = ((states[num_species*j+2]) + (states[num_species*j+4]));
        }
    }
    arma::Row<int> rhs_constr_hyperrec{3, 10, 10, 10, 10};
    arma::Row<double> expansion_factors_hyperrec{0.0, 0.25, 0.25, 0.25, 0.25};
    arma::Row<int> rhs_constr{3, 10, 10, 10, 10, 10, 10};
    arma::Row<double> expansion_factors{0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};

}
