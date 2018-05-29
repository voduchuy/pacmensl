static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <HyperRecOp.h>
#include <Magnus4FSP.h>
#include <FSP.h>

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

/* Stoichiometric matrix of the toggle switch model */
arma::Mat<int> SM{{1, 1, -1, 0, 0, 0},
                  {0, 0, 0,  1, 1, -1}};

const int nReaction = 6;

/* Parameters for the propensity functions */
const double ayx{6.1e-3}, axy{2.6e-3}, nyx{4.1e0}, nxy{3.0e0},
        kx0{6.8e-3}, kx{1.6}, dx{0.00067}, ky0{2.2e-3}, ky{1.7}, dy{3.8e-4};


PetscReal propensity(PetscInt *X, PetscInt k);

arma::Row<PetscReal> t_fun(PetscReal t) {
    //return {kx0, kx, dx, ky0, ky, dy};
    return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
}

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

int main(int argc, char *argv[]) {
    int ierr, myRank;

    /* CME problem sizes */
    int nSpecies = 2; // Number of species
    Row<PetscReal> FSPIncrement({0.25, 0.25}); // Max fraction of states added to each dimension when expanding the FSP
    Row<PetscInt> FSPSize({100, 100}); // Size of the FSP
    PetscReal t_final = 100.0;
    PetscReal fsp_tol = 1.0e-4;
    arma::Mat<PetscInt> init_states(2,1); init_states(0,0) = 0; init_states(1,0) = 0;
    arma::Col<PetscReal> init_prob({1.0});

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);

    MPI_Comm comm = PETSC_COMM_WORLD;

    cme::petsc::FSP my_fsp(comm, init_states, init_prob, SM,
            propensity, t_fun, FSPSize, FSPIncrement, t_final, fsp_tol);
    my_fsp.solve();
    petscvec_to_file(comm, my_fsp.get_P(), "toggle_tv.out");
    my_fsp.destroy();
    ierr = PetscFinalize();
    return ierr;
}

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


void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename) {
    PetscViewer viewer;
    PetscViewerCreate(comm, &viewer);
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    VecView(x, viewer);
    PetscViewerDestroy(&viewer);
}
