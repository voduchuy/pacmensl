//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include "cme_util.h"
#include "MatrixSet.h"
#include "FiniteProblemSolver.h"
#include "CVODEFSP.h"
#include "models/toggle_model.h"

using namespace cme::petsc;

int main(int argc, char *argv[]) {
    PetscInt ierr;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    // Begin PETSC context
    {
        arma::Row<PetscInt> fsp_size = {30, 30};
        FiniteStateSubsetGraph fsp(PETSC_COMM_WORLD);
        fsp.SetSize(fsp_size);
        fsp.SetStoichiometry(toggle_cme::SM);
        fsp.GenerateStatesAndOrdering();
        PetscPrintf(PETSC_COMM_WORLD, "State Subset generated with Graph-partitioned layout.\n");

        MatrixSet A(PETSC_COMM_WORLD);
        A.GenerateMatrices(fsp, toggle_cme::SM, toggle_cme::propensity, toggle_cme::t_fun);

        auto AV = [&A](PetscReal t, Vec x, Vec y) {
            A.Action(t, x, y);
        };


        Vec P;
        VecCreate(PETSC_COMM_WORLD, &P);
        VecSetSizes(P, fsp.GetNumLocalStates() + fsp.GetNumSpecies(), PETSC_DECIDE);
        VecSetFromOptions(P);
        VecSetValue(P, 0, 1.0, INSERT_VALUES);
        VecSetUp(P);
        VecAssemblyBegin(P);
        VecAssemblyEnd(P);

        PetscPrintf(PETSC_COMM_WORLD, "Initial vector set.\n");

        PetscReal fsp_tol = 1.0e-2, t_final = 1000.0;
        CVODEFSP cvode_solver(PETSC_COMM_WORLD, CV_BDF, CV_NEWTON);
        cvode_solver.SetFinalTime(t_final);
        cvode_solver.SetFSPTolerance(fsp_tol);
        cvode_solver.SetInitSolution(&P);
        cvode_solver.SetRHS(AV);
        cvode_solver.SetFiniteStateSubset(&fsp);
        cvode_solver.SetPrintIntermediateSteps(1);
        PetscPrintf(PETSC_COMM_WORLD, "Solver parameters set.\n");
        PetscInt solver_stat = cvode_solver.Solve();
        PetscPrintf(PETSC_COMM_WORLD, "\n Solver returns with status %d and time %.2e \n", solver_stat,
                    cvode_solver.GetCurrentTime());
    }
    //End PETSC context
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    return 0;
}