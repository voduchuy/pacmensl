//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"util/cme_util.h"
#include"FSS/FiniteStateSubset.h"

using namespace cme::parallel;

arma::Mat<PetscInt> SM{
        {1, -1, 0, 0},
        {0, 0,  1, -1}
};

int main(int argc, char *argv[]) {
    int ierr;
    float ver;

    cme::ParaFSP_init(&argc, &argv, help);

    MPI_Comm comm;
    ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    CHKERRQ(ierr);
    {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        arma::Mat<PetscInt> X0(2, 2);
        X0.col(0).fill(0);
        X0.col(1).fill(my_rank + 1);

        FiniteStateSubset fsp(comm, 2);
        fsp.SetStoichiometry(SM);
        fsp.SetInitialStates(X0);

        // Generate a small FSP
        arma::Row<int> fsp_size = {5, 5};
        fsp.SetShapeBounds(fsp_size);
        PetscPrintf(comm, "Initial states:\n");
        PetscPrintf(comm, "State | Petsc ordering \n");
        auto print_states = [](void *data) -> void {
            auto mat_data = (arma::Mat<PetscInt> *) data;
            std::cout << *mat_data;
            return;
        };
        arma::Mat<PetscInt> local_states = fsp.GetLocalStates();
        arma::Row<PetscInt> petsc_indices = fsp.State2Petsc( local_states, false );
        arma::Mat<PetscInt> local_table = arma::join_horiz(local_states.t(), petsc_indices.t());
        cme::sequential_action(comm, print_states, (void *) &local_table);
        fsp.GenerateStatesAndOrdering();
        local_states = fsp.GetLocalStates();
        petsc_indices = fsp.State2Petsc( local_states, false );
        local_table = arma::join_horiz(local_states.t(), petsc_indices.t());
        cme::sequential_action(comm, print_states, (void *) &local_table);

        // Timing for generating a big FSP
        PetscReal t1, t2;
        PetscTime(&t1);
        for (int i{0}; i < 10; ++i) {
            fsp_size+=100;
            fsp.SetShapeBounds(fsp_size);
            fsp.GenerateStatesAndOrdering();
            PetscTime(&t2);
            int nglobal = fsp.GetNumGlobalStates();
            PetscPrintf(comm, "FSP expansion takes %.2f second, global state set size = %d.\n", t2 - t1, nglobal);
        }
        PetscPrintf(comm, "FSP expansion takes %.2f second.\n", t2 - t1);
    }
    CHKERRQ(ierr);
    MPI_Comm_free(&comm);

    cme::ParaFSP_finalize();
}
