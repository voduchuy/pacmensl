//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include <gtest/gtest.h>
#include "Sys.h"
#include "FspMatrixConstrained.h"
#include "CvodeFsp.h"
#include "KrylovFsp.h"
#include"pacmensl_test_env.h"

using namespace pacmensl;

namespace toggle_cme {
/* Stoichiometric matrix of the toggle switch model */
arma::Mat<PetscInt> SM{{1, 1, -1, 0, 0, 0},
                       {0, 0, 0, 1, 1, -1}};

const int nReaction = 6;

/* Parameters for the propensity functions */
const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3.0e0}, nxy{2.1e0}, kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{
    1.6e-2}, dy{3.8e-4};

// Function to constraint the shape of the Fsp
void lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,
                void *args) {

  for (int i{0}; i < num_states; ++i) {
    vals[i * num_constrs]     = states[num_species * i];
    vals[i * num_constrs + 1] = states[num_species * i + 1];
    vals[i * num_constrs + 2] = states[num_species * i] * states[num_species * i + 1];
  }
}

arma::Row<int>    rhs_constr{200, 200, 2000};
arma::Row<double> expansion_factors{0.2, 0.2, 0.2};

// propensity function for toggle
int propensity(const int reaction,
               const int num_species,
               const int num_states,
               const PetscInt *X,
               double *outputs,
               void *args) {
  int (*X_view)[2] = ( int (*)[2] ) X;
  switch (reaction) {
    case 0:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0; }
      break;
    case 1:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0 / (1.0 + ayx * pow(PetscReal(X_view[i][1]), nyx)); }
      break;
    case 2:for (int i{0}; i < num_states; ++i) { outputs[i] = PetscReal(X_view[i][0]); }
      break;
    case 3:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0; }
      break;
    case 4:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0 / (1.0 + axy * pow(PetscReal(X_view[i][0]), nxy)); }
      break;
    case 5:for (int i{0}; i < num_states; ++i) { outputs[i] = PetscReal(X_view[i][1]); }
      break;
    default:return -1;
  }
  return 0;
}

int t_fun(PetscReal t, int n_coefs, double *outputs, void *args) {
  outputs[0] = kx0;
  outputs[1] = kx;
  outputs[2] = dx;
  outputs[3] = ky0;
  outputs[4] = ky;
  outputs[5] = dy;
  return 0;
}
}

class OdeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    // Begin PETSC context
    arma::Row<PetscInt> fsp_size = {100, 100};
    arma::Mat<PetscInt> X0(2, 1);
    X0.col(0).fill(0);
    StateSetConstrained fsp(PETSC_COMM_WORLD);
    fsp.SetShapeBounds(fsp_size);
    fsp.SetStoichiometryMatrix(toggle_cme::SM);
    fsp.SetUp();
    fsp.AddStates(X0);
    fsp.Expand();

    A = new FspMatrixConstrained(PETSC_COMM_WORLD);
    A->GenerateValues(fsp, toggle_cme::SM, toggle_cme::propensity, nullptr, toggle_cme::t_fun, nullptr);
  };

  void TearDown() override { delete A; };

  FspMatrixConstrained *A = nullptr;
};

TEST_F(OdeTest, use_cvode_bdf) {

  auto AV = [&](PetscReal t, Vec x, Vec y) {
    return A->Action(t, x, y);
  };

  Vec P;
  VecCreate(PETSC_COMM_WORLD, &P);
  VecSetSizes(P, A->GetNumLocalRows(), PETSC_DECIDE);
  VecSetFromOptions(P);
  VecSetValue(P, 0, 1.0, INSERT_VALUES);
  VecSetUp(P);
  VecAssemblyBegin(P);
  VecAssemblyEnd(P);

  PetscReal fsp_tol = 1.0e-2, t_final = 100.0;
  CvodeFsp  cvode_solver(PETSC_COMM_WORLD, CV_BDF);
  cvode_solver.SetFinalTime(t_final);
  cvode_solver.SetInitialSolution(&P);
  cvode_solver.SetRhs(AV);
  cvode_solver.SetStatusOutput(0);
  PetscInt solver_stat = cvode_solver.Solve();
  ASSERT_FALSE(solver_stat);

  PetscReal Psum;
  VecSum(P, &Psum);
  ASSERT_LE(Psum, 1.0 + 1.0e-8);
  ASSERT_GE(Psum, 1.0 - 1.0e-8);
  VecDestroy(&P);
}

TEST_F(OdeTest, cvode_handling_bad_mat_vec) {

  auto AV = [&](PetscReal t, Vec x, Vec y) {
    return -1;
  };

  Vec P;
  VecCreate(PETSC_COMM_WORLD, &P);
  VecSetSizes(P, A->GetNumLocalRows(), PETSC_DECIDE);
  VecSetFromOptions(P);
  VecSetValue(P, 0, 1.0, INSERT_VALUES);
  VecSetUp(P);
  VecAssemblyBegin(P);
  VecAssemblyEnd(P);

  PetscReal fsp_tol = 1.0e-2, t_final = 100.0;
  CvodeFsp  cvode_solver(PETSC_COMM_WORLD, CV_BDF);
  cvode_solver.SetFinalTime(t_final);
  cvode_solver.SetInitialSolution(&P);
  cvode_solver.SetRhs(AV);
  cvode_solver.SetStatusOutput(0);
  PetscInt solver_stat = cvode_solver.Solve();
  ASSERT_EQ(solver_stat, -1);

  PetscReal Psum;
  VecSum(P, &Psum);
  ASSERT_LE(Psum, 1.0 + 1.0e-8);
  ASSERT_GE(Psum, 1.0 - 1.0e-8);
  VecDestroy(&P);
}

TEST_F(OdeTest, use_krylov) {

  auto AV = [&](PetscReal t, Vec x, Vec y) {
    return A->Action(t, x, y);
  };

  Vec P;
  VecCreate(PETSC_COMM_WORLD, &P);
  VecSetSizes(P, A->GetNumLocalRows(), PETSC_DECIDE);
  VecSetFromOptions(P);
  VecSetValue(P, 0, 1.0, INSERT_VALUES);
  VecSetUp(P);
  VecAssemblyBegin(P);
  VecAssemblyEnd(P);

  PetscReal fsp_tol = 1.0e-2, t_final = 100.0;
  KrylovFsp krylov_solver(PETSC_COMM_WORLD);

  krylov_solver.SetFinalTime(t_final);
  krylov_solver.SetInitialSolution(&P);
  krylov_solver.SetRhs(AV);
  krylov_solver.SetStatusOutput(0);
  PetscInt solver_stat = krylov_solver.Solve();
  krylov_solver.FreeWorkspace();
  ASSERT_FALSE(solver_stat);

  PetscReal Psum;
  VecSum(P, &Psum);
  ASSERT_LE(Psum, 1.0 + 1.0e-8);
  ASSERT_GE(Psum, 1.0 - 1.0e-8);
  VecDestroy(&P);
}

TEST_F(OdeTest, krylov_handling_bad_mat_vec) {
  int  ierr;
  auto AV = [&](PetscReal t, Vec x, Vec y) {
    return -1;
  };

  Vec P;
  VecCreate(PETSC_COMM_WORLD, &P);
  VecSetSizes(P, A->GetNumLocalRows(), PETSC_DECIDE);
  VecSetFromOptions(P);
  VecSetValue(P, 0, 1.0, INSERT_VALUES);
  VecSetUp(P);
  VecAssemblyBegin(P);
  VecAssemblyEnd(P);

  PetscReal fsp_tol = 1.0e-2, t_final = 100.0;
  KrylovFsp krylov_solver(PETSC_COMM_WORLD);

  ierr = krylov_solver.SetFinalTime(t_final);
  ASSERT_EQ(ierr, 0);
  ierr = krylov_solver.SetInitialSolution(&P);
  ASSERT_EQ(ierr, 0);
  ierr = krylov_solver.SetRhs(AV);
  ASSERT_EQ(ierr, 0);
  ierr                 = krylov_solver.SetStatusOutput(0);
  ASSERT_EQ(ierr, 0);
  PetscInt solver_stat = krylov_solver.Solve();
  ASSERT_EQ(solver_stat, -1);

  VecDestroy(&P);
}