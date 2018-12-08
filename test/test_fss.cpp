//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"models/hog1p_tv_model.h"
#include"cme_util.h"
#include"FiniteStateSubset.h"
#include"FiniteStateSubsetLinear.h"
#include"FiniteStateSubsetParMetis.h"

using namespace cme::petsc;

int main(int argc, char *argv[]) {
    int ierr;
    arma::Row<PetscInt> fsp_size = {1, 1, 1, 1, 1};

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    {
        FiniteStateSubset *fsp = new FiniteStateSubsetLinear(PETSC_COMM_WORLD);

        fsp->SetSize(fsp_size);
        fsp->GenerateStatesAndOrdering();
        arma::Mat<PetscInt> local_states = fsp->GetLocalStates();
        arma::Row<PetscInt> petsc_indices = fsp->State2Petsc(local_states);
        fsp->PrintAO();
        MPI_Barrier(PETSC_COMM_WORLD);

        fsp = new FiniteStateSubsetParMetis(PETSC_COMM_WORLD);
        fsp->SetSize(fsp_size);
        fsp->SetStoichiometry(hog1p_cme::SM);
        fsp->GenerateStatesAndOrdering();
        fsp->PrintAO();
        delete fsp;
    }
    ierr = PetscFinalize();
    CHKERRQ(ierr);
}