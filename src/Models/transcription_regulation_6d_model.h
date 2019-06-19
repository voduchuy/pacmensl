#pragma once

#include <armadillo>
#include <petscmat.h>

namespace six_species_cme {
// stoichiometric matrix of the toggle switch model
    arma::Mat<PetscInt> SM{
            {1, -1, 0, 0,  0,  0,  0,  0,  -2, 2},
            {0, 0,  0, 0,  -1, 1,  -1, 1,  1,  -1},
            {0, 0,  0, 0,  -1, 1,  0,  0,  0,  0},
            {0, 0,  0, 0,  1,  -1, -1, 1,  0,  0},
            {0, 0,  0, 0,  0,  0,  1,  -1, 0,  0},
            {0, 0,  1, -1, 0,  0,  0,  0,  0,  0}
    };

// reaction parameters
    const PetscReal
            Avo = 6.022140857e23,
            c0 = 0.043,
            c1 = 0.0007,
            c2 = 0.078,
            c3 = 0.0039,
            c4 = 0.012e09 / (Avo),
            c5 = 0.4791,
            c6 = 0.00012e09 / (Avo),
            c7 = 0.8765e-11,
            c8 = 0.05e09 / (Avo),
            c9 = 0.5,
            avg_cell_cyc_time = 35 * 60.0;

// propensity function
    PetscReal propensity(PetscInt *X, PetscInt k) {
        switch (k) {
            case 0:
                return c0 * PetscReal(X[5]);
            case 1:
                return c1 * PetscReal(X[0]);
            case 2:
                return c2 * PetscReal(X[3]);
            case 3:
                return c3 * PetscReal(X[5]);
            case 4:
                return PetscReal(X[1]) * PetscReal(X[2]);
            case 5:
                return c5 * PetscReal(X[3]);
            case 6:
                return PetscReal(X[3]) * PetscReal(X[1]);
            case 7:
                return c7 * PetscReal(X[4]);
            case 8:
                return 0.5 * PetscReal(X[0]) * PetscReal(X[0] - 1);
            case 9:
                return c9 * PetscReal(X[1]);
            default:
                return 0.0;
        }
    }

    // Function to constraint the shape of the FSP
    void lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states,
                    int *vals, void *args) {
        for (int j{0}; j < num_states; ++j) {
            for (int i{0}; i < 6; ++i) {
                vals[j * num_constrs + i] = states[num_species * j + i];
            }
        }
    }

    arma::Row<int> rhs_constr{10, 6, 1, 2, 1, 1};
    arma::Row<double> expansion_factors{0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    arma::Row<int> rhs_constr_hyperrec{10, 6, 1, 2, 1, 1};
    arma::Row<double> expansion_factors_hyperrec{0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    // function to compute the time-dependent coefficients of the propensity functions
    arma::Row<PetscReal> t_fun(PetscReal t) {
        arma::Row<PetscReal> u(10, arma::fill::ones);

        PetscReal AV = 6.022140857 * 1.0e8 * pow(2.0, t / avg_cell_cyc_time); // cell's volume
        u(4) = 0.012e09 / AV;
        u(6) = 0.00012e09 / AV;
        u(8) = 0.05e09 / AV;
        return u;
    }

}
