//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include "util/cme_util.h"
#include "Matrix/FspMatrixBase.h"
#include "OdeSolverBase.h"
#include "OdeSolver/CvodeFsp.h"
#include "models/toggle_model.h"

using namespace cme::parallel;

int main(int argc, char *argv[]) {
    PetscInt ierr;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    // Begin PETSC context
    {
        arma::Row<PetscInt> fsp_size = {30, 30};
        arma::Mat<PetscInt> X0(2, 1);
        X0.col(0).fill(0);
        StateSetConstrained fsp(PETSC_COMM_WORLD, 2);
        fsp.SetShapeBounds(fsp_size);
        fsp.SetStoichiometryMatrix(toggle_cme::SM);
        fsp.SetInitialStates(X0);
        fsp.Expand();
        PetscPrintf(PETSC_COMM_WORLD, "State Subset generated with Graph-partitioned layout.\n");

        FspMatrixBase A(PETSC_COMM_WORLD);
        A.generate_values(fsp, toggle_cme::SM, toggle_cme::propensity, toggle_cme::t_fun);

        auto AV = [&A](PetscReal t, Vec x, Vec y) {
            A.action(t, x, y);
        };


        Vec P;
        VecCreate(PETSC_COMM_WORLD, &P);
        VecSetSizes(P, A.get_num_rows_local(), PETSC_DECIDE);
        VecSetFromOptions(P);
        VecSetValue(P, 0, 1.0, INSERT_VALUES);
        VecSetUp(P);
        VecAssemblyBegin(P);
        VecAssemblyEnd(P);

        PetscPrintf(PETSC_COMM_WORLD, "Initial vector set.\n");

        PetscReal fsp_tol = 1.0e-2, t_final = 1000.0;
        CvodeFsp cvode_solver( PETSC_COMM_WORLD, CV_BDF );
        cvode_solver.set_final_time(t_final);
        cvode_solver.set_initial_solution(&P);
        cvode_solver.set_rhs(AV);
        cvode_solver.set_print_intermediate(1);
        PetscPrintf(PETSC_COMM_WORLD, "Solver parameters set.\n");
        PetscInt solver_stat = cvode_solver.solve();
        PetscPrintf(PETSC_COMM_WORLD, "\n Solver returns with status %d and time %.2e \n", solver_stat,
                    cvode_solver.get_current_time());
    }
    //End PETSC context
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    return 0;
}