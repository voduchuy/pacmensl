//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"models/hog1p_5d_model.h"
#include"cme_util.h"
#include"FiniteStateSubset.h"
#include"FiniteStateSubsetNaive.h"
#include"FiniteStateSubsetGraph.h"
#include"FiniteStateSubsetHyperGraph.h"

using namespace cme::petsc;

int main(int argc, char *argv[]) {
    int ierr;
    arma::Row<PetscInt> fsp_size = {1, 1, 1, 1, 1};
    float ver;

    MPI_Init(&argc, &argv);
    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    Zoltan_Initialize(argc, argv, &ver);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    CHKERRQ(ierr);
    {
        FiniteStateSubset *fsp = new FiniteStateSubsetNaive(comm);

        fsp->SetSize(fsp_size);
        fsp->GenerateStatesAndOrdering();
        arma::Mat<PetscInt> local_states = fsp->GetLocalStates();
        arma::Row<PetscInt> petsc_indices = fsp->State2Petsc(local_states);
        fsp->PrintAO();
        delete fsp;

        MPI_Barrier(comm);

        fsp = new FiniteStateSubsetGraph(comm);
        fsp->SetSize(fsp_size);
        fsp->SetStoichiometry(hog1p_cme::SM);
        fsp->GenerateStatesAndOrdering();
        fsp->PrintAO();
        delete fsp;

        fsp = new FiniteStateSubsetHyperGraph(comm);
        fsp->SetSize(fsp_size);
        fsp->SetStoichiometry(hog1p_cme::SM);
        fsp->GenerateStatesAndOrdering();
        fsp->PrintAO();
        delete fsp;
    }
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    MPI_Comm_free(&comm);
    MPI_Finalize();
}