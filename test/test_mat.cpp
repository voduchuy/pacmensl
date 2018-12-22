//
// Created by Huy Vo on 12/4/18.
//
//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"models/toggle_model.h"
#include"cme_util.h"
#include"FiniteStateSubset.h"
#include"FiniteStateSubsetNaive.h"
#include"FiniteStateSubsetGraph.h"
#include"FiniteStateSubsetHyperGraph.h"
#include"FiniteStateSubsetHierarch.h"
#include"FiniteStateSubsetRCB.h"
#include"MatrixSet.h"

using namespace cme::parallel;

int main(int argc, char *argv[]) {
    int ierr;
    arma::Row<PetscInt> fsp_size = {3, 3};

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Read options for fsp
    char opt[100];
    PetscBool opt_set;
    PartitioningType fsp_par_type = Naive;
    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        fsp_par_type = str2part(std::string(opt));
        PetscPrintf(PETSC_COMM_WORLD, "Partitioning with option %s \n", opt);
    }

    double Q_sum;
    {
        FiniteStateSubset *fsp;
        switch (fsp_par_type){
            case Graph:
                fsp = new FiniteStateSubsetGraph(PETSC_COMM_WORLD);
                break;
            case RCB:
                fsp = new FiniteStateSubsetRCB(PETSC_COMM_WORLD);
                break;
            case HyperGraph:
                fsp = new FiniteStateSubsetHyperGraph(PETSC_COMM_WORLD);
                break;
            case Naive:
                fsp = new FiniteStateSubsetNaive(PETSC_COMM_WORLD);
                break;
            case Hierarch:
                fsp = new FiniteStateSubsetHierarch(PETSC_COMM_WORLD);
                break;
            default:
                fsp = new FiniteStateSubsetNaive(PETSC_COMM_WORLD);
                break;
        }
        fsp->SetSize(fsp_size);
        fsp->SetStoichiometry(toggle_cme::SM);
        fsp->GenerateStatesAndOrdering();
        PetscPrintf(PETSC_COMM_WORLD, "State Subset generated with hypergraph partitioning.\n");

        MatrixSet A(PETSC_COMM_WORLD);
        A.GenerateMatrices(*fsp, toggle_cme::SM, toggle_cme::propensity, toggle_cme::t_fun);

        Vec P, Q;
        VecCreate(PETSC_COMM_WORLD, &P);
        VecSetSizes(P, fsp->GetNumLocalStates() + fsp->GetNumSpecies(), PETSC_DECIDE);
        VecSetFromOptions(P);
        VecSet(P, 1.0);
        VecSetUp(P);
        VecDuplicate(P, &Q);
        VecSetUp(Q);

        A.Action(1.0, P, Q);
        VecView(Q, PETSC_VIEWER_STDOUT_WORLD);


        VecSum(Q, &Q_sum);
        PetscPrintf(PETSC_COMM_WORLD, "Q_sum = %.2f \n", Q_sum);

        A.Destroy();
        delete fsp;
    }
    ierr = PetscFinalize();
    CHKERRQ(ierr);
//    assert(std::abs(Q_sum) <= 1.0e-14);
    return 0;
}
