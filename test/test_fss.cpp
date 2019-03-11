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

arma::Mat<PetscInt> SM {
        {  1,  -1,  0,  0},
        {  0,  0,   1, -1}
};

void sequential_action(MPI_Comm comm, std::function<void (void*)> action, void* data){
    int my_rank, comm_size;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &comm_size);

    if (comm_size == 1){
        action(data);
        return;
    }

    int print;
    MPI_Status status;
    if (my_rank == 0){
        std::cout << "Processor " << my_rank << "\n";
        action(data);
        MPI_Send(&print, 1, MPI_INT, my_rank+1, 1, comm);
    }
    else{
        MPI_Recv(&print, 1, MPI_INT, my_rank-1, 1, comm, &status);
        std::cout << "Processor " << my_rank << "\n";
        action(data);
        if (my_rank < comm_size-1){
            MPI_Send(&print, 1, MPI_INT, my_rank+1, 1, comm);
        }
    }
    MPI_Barrier(comm);
}

int main(int argc, char *argv[]) {
    int ierr;
    arma::Row<double> fsp_size = {4, 4};
    float ver;

    cme::ParaFSP_init(&argc, &argv, help);

    MPI_Comm comm;
    ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    CHKERRQ(ierr);
    {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        arma::Mat<PetscInt> X0(2, 2);
        X0.col(0).fill(my_rank);
        X0.col(1).fill(my_rank+1);

        FiniteStateSubset fsp(comm, 2);
        fsp.SetShapeBounds(fsp_size);
        fsp.SetStoichiometry(SM);
        fsp.SetInitialStates(X0);

        PetscPrintf(comm, "Initial states:\n");
        PetscPrintf(comm, "State | Petsc ordering \n");
        auto print_states = [] (void* data) -> void {
            auto mat_data = (arma::Mat<PetscInt>*) data;
            std::cout << *mat_data;
            return;
        };
        arma::Mat<PetscInt> local_states = fsp.GetLocalStates();
        arma::Row<PetscInt> petsc_indices = fsp.State2Petsc(local_states);
        arma::Mat<PetscInt> local_table = arma::join_horiz(local_states.t(), petsc_indices.t());
        sequential_action(comm, print_states, (void*) &local_table);
        fsp.GenerateStatesAndOrdering();
        local_states = fsp.GetLocalStates();
        petsc_indices = fsp.State2Petsc(local_states);
        local_table = arma::join_horiz(local_states.t(), petsc_indices.t());
        sequential_action(comm, print_states, (void*) &local_table);
    }
    CHKERRQ(ierr);
    MPI_Comm_free(&comm);

    cme::ParaFSP_finalize();
}
