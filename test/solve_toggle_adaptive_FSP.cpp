static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <HyperRecOp.h>
#include <Magnus4FSP.h>
<<<<<<< HEAD
#include <FSP.h>
=======
>>>>>>> abc229acd3e18bd997ea308366ce7ed66cae8cb9

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
<<<<<<< HEAD
    //return {kx0, kx, dx, ky0, ky, dy};
    return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
=======
    return {kx0, kx, dx, ky0, ky, dy};
    //return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
>>>>>>> abc229acd3e18bd997ea308366ce7ed66cae8cb9
}

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

int main(int argc, char *argv[]) {
    int ierr, myRank;

    /* CME problem sizes */
    int nSpecies = 2; // Number of species
    Row<PetscReal> FSPIncrement({0.25, 0.25}); // Max fraction of states added to each dimension when expanding the FSP
    Row<PetscInt> FSPSize({100, 100}); // Size of the FSP
    PetscReal t_final = 100.0;
<<<<<<< HEAD
    PetscReal fsp_tol = 1.0e-4;
    arma::Mat<PetscInt> init_states(2,1); init_states(0,0) = 0; init_states(1,0) = 0;
    arma::Col<PetscReal> init_prob({1.0});
=======
>>>>>>> abc229acd3e18bd997ea308366ce7ed66cae8cb9

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);

    MPI_Comm comm = PETSC_COMM_WORLD;
<<<<<<< HEAD

    cme::petsc::FSP my_fsp(comm, init_states, init_prob, SM,
            propensity, t_fun, FSPSize, FSPIncrement, t_final, fsp_tol);
    my_fsp.solve();
    petscvec_to_file(comm, my_fsp.get_P(), "toggle_tv.out");
    my_fsp.destroy();
=======
    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    /* Test the action of the operator */
    cme::petsc::HyperRecOp A(comm, FSPSize, SM, propensity, t_fun);
    PetscPrintf(comm, "Matrix object created. \n");

    Vec P0, P;
    VecCreate(comm, &P0);
    VecCreate(comm, &P);

    VecSetSizes(P0, PETSC_DECIDE, arma::prod(FSPSize + 1) + FSPSize.n_elem);
    VecSetType(P0, VECMPI);
    VecSetUp(P0);
    VecSet(P0, 0.0);
    VecSetValue(P0, 0, 1.0, INSERT_VALUES);
    VecAssemblyBegin(P0);
    VecAssemblyEnd(P0);

    VecDuplicate(P0, &P);
    VecCopy(P0, P);

    PetscPrintf(comm, "P0 and P created. \n");

    auto tmatvec = [&A](PetscReal t, Vec x, Vec y) { A(t).action(x, y); };
    cme::petsc::Magnus4FSP my_magnus(comm, 0.0e0, t_final, tmatvec, P, FSPSize.n_elem, 1.0e-4);

    double t1 = MPI_Wtime();
    PetscReal t = 0.0e0;
    while (t < t_final) {
        my_magnus.solve();
        t = my_magnus.t_now;
        // Expand the FSP if the solver halted prematurely
        if (t < t_final) {
            arma::Row<PetscInt> FSPSize_old {FSPSize};
            for (size_t i{0}; i < my_magnus.to_expand.n_elem; ++i)
            {
                if (my_magnus.to_expand(i) == 1)
                {
                    FSPSize(i) = (PetscInt) std::ceil(((double) FSPSize(i)) * (FSPIncrement(i) + 1.0e0));
                }
            }
            std::cout << FSPSize << std::endl;

            // Generate the expanded matrices
            A.destroy();
            A.generate_matrices(FSPSize, SM, propensity);

            // Generate the expanded vector and scatter forward the current solution
            Vec Pnew;
            VecCreate(comm, &Pnew);
            VecSetSizes(Pnew, PETSC_DECIDE, arma::prod(FSPSize + 1) + FSPSize.n_elem);
            VecSetType(Pnew, VECMPI);
            VecSetUp(Pnew);
            VecSet(Pnew, 0.0);

            IS is_old;

            PetscInt istart, iend, n_old, n_new;
            ierr = VecGetSize(Pnew, &n_new); CHKERRQ(ierr);
            ierr = VecGetSize(P, &n_old); CHKERRQ(ierr);

            ierr = VecGetOwnershipRange(P, &istart, &iend); CHKERRQ(ierr);
            arma::Row<PetscInt> indices_sinks; indices_sinks.set_size(0);
            if (iend >= n_old)
            {
                indices_sinks.set_size(my_magnus.n_sinks);

                for (PetscInt i {0}; i < my_magnus.n_sinks; ++i)
                {
                    indices_sinks(i) = n_new + (i-my_magnus.n_sinks);
                }
            }
            iend = std::min(iend, arma::prod(FSPSize_old+1));
            arma::Row<PetscInt> indices_new = arma::linspace<arma::Row<PetscInt>>(istart, iend - 1, iend - istart); CHKERRQ(ierr);
            arma::Mat<PetscInt> my_X = cme::ind2sub_nd(FSPSize_old, indices_new);
            indices_new = cme::sub2ind_nd(FSPSize, my_X);
            indices_new = arma::join_horiz(indices_new, indices_sinks);

            ierr = ISCreateGeneral(comm, (PetscInt) indices_new.n_elem, indices_new.begin(), PETSC_COPY_VALUES, &is_old); CHKERRQ(ierr);

            VecScatter scatter;
            ierr = VecScatterCreate(P, NULL, Pnew, is_old, &scatter); CHKERRQ(ierr);
            ierr = VecScatterBegin(scatter, P, Pnew, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
            ierr = VecScatterEnd(scatter, P, Pnew, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

            ierr = VecDestroy(&P); CHKERRQ(ierr);
            ierr = VecDuplicate(Pnew, &P); CHKERRQ(ierr);
            ierr = VecSwap(P, Pnew); CHKERRQ(ierr);
            ierr = VecScatterDestroy(&scatter); CHKERRQ(ierr);
            ierr = VecDestroy(&Pnew); CHKERRQ(ierr);
            std::cout << "Update Magnus4 object \n";
            // Reset the Magnus4 object
            my_magnus.destroy();
            my_magnus.update_vector(P, tmatvec);
        }
    }


    std::cout << FSPSize << std::endl;

    double t2 = MPI_Wtime() - t1;
    PetscPrintf(comm, "Solver time %2.4f \n", t2);

    PetscPrintf(comm, "Solver stopped at t = %2.4e \n", my_magnus.t_now);

    petscvec_to_file(comm, P, "toggle_tv.out");

    my_magnus.destroy();
    A.destroy();
    ierr = VecDestroy(&P);
    CHKERRQ(ierr);
    ierr = VecDestroy(&P0);
    CHKERRQ(ierr);
    ierr = VecDestroy(&P);
    CHKERRQ(ierr);
    ierr = VecDestroy(&P0);
    CHKERRQ(ierr);
>>>>>>> abc229acd3e18bd997ea308366ce7ed66cae8cb9
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
