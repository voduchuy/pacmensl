//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test memory scalable AO.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<vector>

int main(int argc, char *argv[]) {
    PetscInt ierr;

    PetscInt n_global = 16;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);

    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscMPIInt num_procs, my_rank;
    MPI_Comm_size(comm, &num_procs);

    PetscLayout layout;

    PetscInt local_size;
    PetscInt start, end;

    PetscMPIInt rank;
    MPI_Comm_rank(comm, &rank);

    ierr = PetscLayoutCreate(comm, &layout);
    CHKERRABORT(comm, ierr);
    ierr = PetscLayoutSetSize(layout, n_global);
    CHKERRABORT(comm, ierr);
    ierr = PetscLayoutSetLocalSize(layout, PETSC_DECIDE);
    CHKERRABORT(comm, ierr);
    ierr = PetscLayoutSetUp(layout);
    CHKERRABORT(comm, ierr);
    ierr = PetscLayoutGetLocalSize(layout, &local_size);
    CHKERRABORT(comm, ierr);
    ierr = PetscLayoutGetRange(layout, &start, &end);
    CHKERRABORT(comm, ierr);


    std::vector<PetscInt> app_indices;
    std::vector<PetscInt> petsc_indices;
    app_indices.resize(local_size);
    petsc_indices.resize(local_size);
    // Add values for local indices for usual states
    for (PetscInt i = 0; i < local_size; ++i) {
        app_indices[i] = start + i;
        petsc_indices[i] = end -1 - i;
    }

    // Create the AO object that maps from lexicographic ordering to Petsc Vec ordering
    AO app2petsc;
    IS app_is, petsc_is;
    ISCreateGeneral(comm, local_size, &app_indices[0], PETSC_COPY_VALUES, &app_is);
    ISCreateGeneral(comm, local_size, &petsc_indices[0], PETSC_COPY_VALUES, &petsc_is);
    AOCreate(comm, &app2petsc);
    AOSetIS(app2petsc, app_is, petsc_is);
    AOSetType(app2petsc, AOMEMORYSCALABLE);
    AOSetFromOptions(app2petsc);
    CHKERRABORT(comm, ISDestroy(&app_is));
    CHKERRABORT(comm, ISDestroy(&petsc_is));

    AOView(app2petsc, PETSC_VIEWER_STDOUT_WORLD);

    // Test AOApplicationToPetsc
    const PetscInt n_loc = 8;
    std::vector<PetscInt> ia(n_loc);
    if (rank == 0) {
        ia = {0, -1, 1, 2, -1, 4, 5, 6};
    } else {
        ia = {-1, 8, 9, 10, -1, 12, 13, 14};
    }
    std::vector<PetscInt> ia0 = ia;
    AOApplicationToPetsc(app2petsc, n_loc, &ia[0]);

    for (auto i{0}; i < n_loc; ++i) {
        printf("proc = %d : %d -> %d \n", rank, ia0[i], ia[i]);
    }

//    Correct result for 2 processes should be:
//    proc = 0 : 0 -> 7
//    proc = 0 : -1 -> -1
//    proc = 0 : 1 -> 6
//    proc = 0 : 2 -> 5
//    proc = 0 : -1 -> -1
//    proc = 0 : 4 -> 3
//    proc = 0 : 5 -> 2
//    proc = 0 : 6 -> 1
//    proc = 1 : -1 -> -1
//    proc = 1 : 8 -> 15
//    proc = 1 : 9 -> 14
//    proc = 1 : 10 -> 13
//    proc = 1 : -1 -> -1
//    proc = 1 : 12 -> 11
//    proc = 1 : 13 -> 10
//    proc = 1 : 14 -> 9

    // AO maps correctly if there are no negative entries
//    if (rank == 0) {
//        ia = {0, 0, 1, 2, 0, 4, 5, 6};
//    } else {
//        ia = {0, 8, 9, 10, 0, 12, 13, 14};
//    }
//    ia0 = ia;
//    AOApplicationToPetsc(app2petsc, n_loc, &ia[0]);
//
//    for (auto i{0}; i < n_loc; ++i) {
//        printf("proc = %d : %d -> %d \n", rank, ia0[i], ia[i]);
//    }

    ierr = PetscLayoutDestroy(&layout);
    CHKERRABORT(comm, ierr);
    ierr = PetscFinalize();
    CHKERRQ(ierr);
}