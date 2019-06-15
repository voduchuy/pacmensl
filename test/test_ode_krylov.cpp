//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to Krylov integrator for solving the CME of the toggle model.\n\n";

#include "cme_util.h"
#include "FspMatrixBase.h"
#include "KrylovFsp.h"
#include "Models/toggle_model.h"

using namespace pecmeal;

int main(int argc, char *argv[]) {
  //PECMEAL parallel environment object, must be created before using other PECMEAL's functionalities
  pecmeal::Environment my_env(&argc, &argv, help);

  PetscInt ierr;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

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
  KrylovFsp krylov_solver(PETSC_COMM_WORLD);

  krylov_solver.SetFinalTime(t_final);
  krylov_solver.set_initial_solution(&P);
  krylov_solver.set_rhs(AV);
  krylov_solver.set_print_intermediate(1);
  PetscPrintf(PETSC_COMM_WORLD, "Solver parameters set.\n");
  PetscInt solver_stat = krylov_solver.Solve();
  PetscPrintf(PETSC_COMM_WORLD, "\n Solver returns with status %d and time %.2e \n", solver_stat,
              krylov_solver.GetCurrentTime());
  krylov_solver.FreeWorkspace();

  PetscReal Psum;
  VecSum(P, &Psum);
  PetscPrintf(PETSC_COMM_WORLD, "Psum = %.2e \n", Psum);
  VecDestroy(&P);
  return 0;
}