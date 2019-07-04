//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test suite for StationaryFspSolverMultiSinks object.\n\n";

#include <gtest/gtest.h>
#include "pacmensl_all.h"
#include "StationaryFspSolverMultiSinks.h"
#include "pacmensl_test_env.h"

using namespace pacmensl;

class BirthDeathTest : public ::testing::Test
{
 protected:
  BirthDeathTest() {}

  void SetUp() override
  {

    auto propensity =
             [&](int reaction, int num_species, int num_states, const int *state, PetscReal *output, void *args) {
               if (reaction == 0)
               {
                 for (int i{0}; i < num_states; ++i)
                 {
                   output[i] = 1.0;
                 }
                 return 0;
               } else
               {
                 for (int i{0}; i < num_states; ++i)
                 {
                   output[i] = state[i];
                 }
                 return 0;
               }
             };

    auto t_fun = [&](double t, int num_coefs, double *outputs, void *args) {
      outputs[0] = lambda;
      outputs[1] = gamma;
      return 0;
    };

    bd_model = Model(stoich_matrix,
                     t_fun,
                     propensity,
                     nullptr,
                     nullptr);
  }

  void TearDown() override
  {

  }

  ~BirthDeathTest() {}

  Model                bd_model;
  PetscReal            lambda            = 2.0;
  PetscReal            gamma             = 1.0;
  arma::Mat<int>       stoich_matrix     = {1, -1};
  arma::Mat<int>       x0                = {0};
  arma::Col<PetscReal> p0                = {1.0};
  arma::Row<int>       fsp_size          = {5};
  arma::Row<PetscReal> expansion_factors = {0.1};
  PetscReal            fsp_tol{1.0e-10};
};

TEST_F(BirthDeathTest, test_solve)
{
  PetscInt             ierr;
  PetscReal            stmp;
  DiscreteDistribution p_stationary;

  StationaryFspSolverMultiSinks fsp(PETSC_COMM_WORLD);

  ierr = fsp.SetModel(bd_model);
  ASSERT_FALSE(ierr);
  ierr = fsp.SetInitialBounds(fsp_size);
  ASSERT_FALSE(ierr);
  ierr = fsp.SetExpansionFactors(expansion_factors);
  ASSERT_FALSE(ierr);
  ierr = fsp.SetInitialDistribution(x0, p0);
  ASSERT_FALSE(ierr);

  p_stationary = fsp.Solve(fsp_tol);
  fsp.ClearState();

  VecView(p_stationary.p_, PETSC_VIEWER_STDOUT_WORLD);
  // Check that the solution is close to exact solution
  stmp        = 0.0;
  PetscReal *p_dat;
  int       num_states;
  p_stationary.GetProbView(num_states, p_dat);
  PetscReal pdf;
  int       n;
  for (int  i = 0; i < num_states; ++i)
  {
    n   = p_stationary.states_(0, i);
    pdf = exp(-(lambda/gamma)) * pow(lambda/gamma, double(n)) / tgamma(n + 1);
    stmp += abs(p_dat[i] - pdf);
  }
  p_stationary.RestoreProbView(p_dat);
  MPI_Allreduce(&stmp, MPI_IN_PLACE, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
  ASSERT_LE(stmp, 1.0e-8);
}
