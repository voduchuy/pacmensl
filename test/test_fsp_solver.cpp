//
// Created by Huy Vo on 12/6/18.
//

//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include "pacmensl_all.h"
#include "FspSolverBase.h"
#include "toggle_model.h"

using namespace pacmensl;

int main(int argc, char *argv[]) {
  Environment my_env(&argc, &argv, help);

  PetscInt ierr;
  PetscReal stmp;
  DiscreteDistribution p_final_bdf, p_final_krylov;
  std::vector<DiscreteDistribution> p_snapshots_bdf, p_snapshots_krylov;
  std::vector<PetscReal> tspan;
  Vec q;

  std::string model_name = "toggle_switch";

  PetscReal t_final = 100.0, fsp_tol = 1.0e-6;
  arma::Mat<PetscInt> X0 = {0, 0};
  X0 = X0.t();
  arma::Col<PetscReal> p0 = {1.0};

  // Get processor rank and number of processors
  PetscMPIInt rank, num_procs;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);

  Model toggle_model(toggle_cme::SM, toggle_cme::t_fun, toggle_cme::propensity);
  arma::Row<int> fsp_size = {100, 100};
  arma::Row<PetscReal> expansion_factors = {0.25, 0.25};

  tspan = arma::conv_to<std::vector<PetscReal>>::from(arma::linspace<arma::Row<PetscReal>>(0.0, t_final, 10));

  FspSolverBase fsp(PETSC_COMM_WORLD);

  fsp.SetModel(toggle_model);
  fsp.SetInitialBounds(fsp_size);
  fsp.SetExpansionFactors(expansion_factors);
  fsp.SetVerbosity(0);
  fsp.SetInitialDistribution(X0, p0);

  fsp.SetOdesType(CVODE_BDF);
  fsp.SetUp();
  p_final_bdf = fsp.Solve(t_final, fsp_tol);
  p_snapshots_bdf = fsp.SolveTspan(tspan, fsp_tol);

  fsp.SetOdesType(KRYLOV);
  fsp.SetUp();
  p_final_krylov = fsp.Solve(t_final, fsp_tol);
  p_snapshots_krylov = fsp.SolveTspan(tspan, fsp_tol);

  ierr = VecDuplicate(p_final_bdf.p, &q); CHKERRQ(ierr);

  ierr = VecCopy(p_final_bdf.p, q); CHKERRQ(ierr);
  ierr = VecAXPY(q, -1.0, p_final_krylov.p); CHKERRQ(ierr);
  ierr = VecNorm(q, NORM_1, &stmp); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Final solution Krylov - BDF = %.2e \n", stmp);

  for (int i{0}; i < tspan.size(); ++i){
    ierr = VecCopy(p_snapshots_bdf[i].p, q); CHKERRQ(ierr);
    ierr = VecAXPY(q, -1.0, p_snapshots_krylov[i].p); CHKERRQ(ierr);
    ierr = VecNorm(q, NORM_1, &stmp); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Final solution at t = %.2e Krylov - BDF = %.2e \n", tspan.at(i), stmp);

    ierr = VecSum(p_final_bdf.p, &stmp); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Sum(p_final_bdf) = %.2e \n", stmp);

    ierr = VecSum(p_final_krylov.p, &stmp); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Sum(p_final_krylov) = %.2e \n", stmp);
  }

  ierr = VecDestroy(&q); CHKERRQ(ierr);
  return 0;
}