//
// Created by Huy Vo on 5/17/18.
//
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscviewer.h>

#include <armadillo>

int main(int argc, char *argv[]) {
    PetscInt ierr, n1{10}, n2{30};
    IS is_x, is_y;
    Vec x, y;
    VecScatter scatter;

    arma::Row<PetscInt> indices_x, indices_y;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, (char *) 0);
    CHKERRQ(ierr);
    MPI_Comm comm{PETSC_COMM_WORLD};

    /* Create x with all ones */
    ierr = VecCreate(comm, &x);
    CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);
    CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n1);
    CHKERRQ(ierr);
    ierr = VecSetUp(x);
    CHKERRQ(ierr);

    ierr = VecSet(x, 1.0e0);
    CHKERRQ(ierr);

    /* Create y with all zeros */
    ierr = VecCreate(comm, &y);
    CHKERRQ(ierr);
    ierr = VecSetFromOptions(y);
    CHKERRQ(ierr);
    ierr = VecSetSizes(y, PETSC_DECIDE, n2);
    CHKERRQ(ierr);
    ierr = VecSetUp(y);
    CHKERRQ(ierr);

    ierr = VecSet(y, 0.0e0);
    CHKERRQ(ierr);

    PetscPrintf(comm, "Vectors created. \n");
    /* Create index set for x */
    PetscInt istart, iend;
    ierr = VecGetOwnershipRange(x, &istart, &iend);
    indices_x = arma::linspace<arma::Row<PetscInt>>(istart, iend - 1, iend - istart);
    ierr = ISCreateGeneral(comm, indices_x.n_elem, indices_x.begin(), PETSC_COPY_VALUES, &is_x);
    PetscPrintf(comm, "IS_X created. \n");

    /* Create index set for y */
    ierr = VecGetOwnershipRange(y, &istart, &iend);
    indices_y = arma::linspace<arma::Row<PetscInt>>(istart, iend - 1, iend - istart);
    ierr = ISCreateGeneral(comm, indices_y.n_elem, indices_y.begin(), PETSC_COPY_VALUES, &is_y);
    PetscPrintf(comm, "IS_Y created. \n");

    /* Create vecscatter context and execute the scattering */
    ierr = VecScatterCreate(x, is_x, y, is_x, &scatter);
    ierr = VecScatterBegin(scatter, x, y, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecScatterEnd(scatter, x, y, INSERT_VALUES, SCATTER_FORWARD);
    ierr = VecScatterDestroy(&scatter);

    /* Let's see what values y get */
    ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD);

    ierr = ISDestroy(&is_y);
    ierr = ISDestroy(&is_x);
    ierr = VecDestroy(&x);
    ierr = VecDestroy(&y);
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    return ierr;
}
