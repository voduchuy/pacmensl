/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
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
                     nullptr,
                     std::vector<int>({0,1}));
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
  PetscReal            fsp_tol{1.0e-18};
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
  ierr = fsp.SetVerbosity(2);

  p_stationary = fsp.Solve(fsp_tol);
  fsp.ClearState();

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
    pdf = exp(-(lambda / gamma)) * pow(lambda / gamma, double(n)) / tgamma(n + 1);
    stmp += abs(p_dat[i] - pdf);
  }
  p_stationary.RestoreProbView(p_dat);
  MPI_Allreduce(&stmp, MPI_IN_PLACE, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
  ASSERT_LE(stmp, 1.0e-8);
}

TEST(bursting_stationary, solve)
{
  int ierr;

  arma::Mat<int> stoich_matrix = {{-1, 1, 0}, {1, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  stoich_matrix = stoich_matrix.t();
  arma::Mat<int>       x0                = {2, 0, 0};
  x0 = x0.t();
  arma::Col<PetscReal> p0                = {1.0};
  arma::Row<int>       fsp_size          = {2, 2, 10};
  arma::Row<PetscReal> expansion_factors = {0.1, 0.1, 0.1};


  double gamma{1.0}, k_r{10}, k_on{0.01}, k_off{0.05};
  auto   propensity                      =
             [&](int reaction, int num_species, int num_states, const int *state, PetscReal *output, void *args) {
               const int (*X)[3] = ( const int (*)[3] ) state;
               if (reaction == 0)
               {
                 for (int i{0}; i < num_states; ++i)
                 {
                   output[i] = X[i][0];
                 }

               } else if (reaction == 1)
               {
                 for (int i{0}; i < num_states; ++i)
                 {
                   output[i] = X[i][1];
                 }
               } else if (reaction == 2)
               {
                 for (int i{0}; i < num_states; ++i)
                 {
                   output[i] = X[i][1];
                 }
               } else
               {
                 for (int i{0}; i < num_states; ++i)
                 {
                   output[i] = X[i][2];
                 }
               }
               return 0;
             };

  auto t_fun = [&](double t, int num_coefs, double *outputs, void *args) {
    outputs[0] = k_on;
    outputs[1] = k_off;
    outputs[2] = k_r;
    outputs[3] = gamma;
    return 0;
  };

  Model model = Model(stoich_matrix,
                      t_fun,
                      propensity,
                      nullptr,
                      nullptr);

  StationaryFspSolverMultiSinks fsp(PETSC_COMM_WORLD);

  ierr = fsp.SetModel(model);
  ASSERT_FALSE(ierr);
  ierr = fsp.SetInitialBounds(fsp_size);
  ASSERT_FALSE(ierr);
  ierr = fsp.SetExpansionFactors(expansion_factors);
  ASSERT_FALSE(ierr);
  ierr = fsp.SetInitialDistribution(x0, p0);
  ASSERT_FALSE(ierr);

  DiscreteDistribution p_stationary = fsp.Solve(1.0e-4);
  fsp.ClearState();

  double stmp;
  VecSum(p_stationary.p_, &stmp);
  PetscPrintf(PETSC_COMM_WORLD, "%.2e", stmp);
}