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
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<gtest/gtest.h>
#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"Sys.h"
#include"FspMatrixConstrained.h"
#include"SensFspMatrix.h"
#include"pacmensl_test_env.h"

using namespace pacmensl;

// Test Fixture: 1-d discrete random walk model
class SensMatrixTest : public ::testing::Test {
 protected:

  SensMatrixTest() {}

  void SetUp() override {
    fsp_size = arma::Row<int>({12});
    t_fun    = [&](double t, int num_coefs, double *outputs, void *args) {
      outputs[0] = rate_right;
      outputs[1] = rate_left;
      return 0;
    };

    d_t_fun1 = [&](double t, int num_coefs, double *outputs, void *args) {
      outputs[0] = 1.0;
      return 0;
    };

    d_t_fun2 = [&](double t, int num_coefs, double *outputs, void *args) {
      outputs[1] = 1.0;
      return 0;
    };

    propensity = [&](const int reaction, const int num_species, const int num_states, const int *X,
                     double *outputs, void *args) {
      switch (reaction) {
        case 0:
          for (int i{0}; i < num_states; ++i) {
            outputs[i] = 1.0;
          }
          break;
        case 1:
          for (int i{0}; i < num_states; ++i) {
            outputs[i] = (X[i] > 0);
          }
          break;
        default:return -1;
      }
      return 0;
    };


    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    int              ierr;
    // Read options for state_set_
    char             opt[100];
    PetscBool        opt_set;
    PartitioningType fsp_par_type = PartitioningType::GRAPH;
    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    ASSERT_FALSE(ierr);

    arma::Mat<PetscInt> X0(1, 1);
    X0.fill(0);
    X0(0) = 0;
    state_set = std::make_shared<StateSetConstrained>(PETSC_COMM_WORLD);
    ierr      = state_set->SetStoichiometryMatrix(stoichiometry);
    ASSERT_FALSE(ierr);
    ierr = state_set->SetShapeBounds(fsp_size);
    ASSERT_FALSE(ierr);
    ierr = state_set->SetUp();
    ASSERT_FALSE(ierr);
    ierr = state_set->AddStates(X0);
    ASSERT_FALSE(ierr);
    ierr = state_set->Expand();
    ASSERT_FALSE(ierr);

    smodel = pacmensl::SensModel(stoichiometry,
        std::vector<int>({0,1}),
        t_fun,
        propensity,
        std::vector<pacmensl::TcoefFun>({d_t_fun1, d_t_fun2}),
        std::vector<pacmensl::PropFun>({propensity, propensity}),
        std::vector<int>({0,1}),
        std::vector<int>({0,1,2})
        );
  };

  void TearDown() override {
  };

  std::shared_ptr<StateSetConstrained> state_set;
  arma::Row<int>                       fsp_size;
  const double                         rate_right = 2.0, rate_left = 3.0;
  const arma::Mat<int>                 stoichiometry{1, -1};
  pacmensl::TcoefFun                   t_fun;
  pacmensl::TcoefFun                   d_t_fun1, d_t_fun2;
  pacmensl::PropFun                    propensity;

  pacmensl::SensModel smodel;
};

TEST_F(SensMatrixTest, mat_base_generation) {
  int ierr;

  double Q_sum;

  pacmensl::SensFspMatrix<FspMatrixBase> A(PETSC_COMM_WORLD);
  ierr = A.GenerateValues(*state_set, smodel);
  ASSERT_FALSE(ierr);

  Vec P, Q;
  ierr = VecCreate(PETSC_COMM_WORLD, &P);
  ASSERT_FALSE(ierr);
  ierr = VecSetSizes(P, state_set->GetNumLocalStates(), PETSC_DECIDE);
  ASSERT_FALSE(ierr);
  ierr = VecSetFromOptions(P);
  ASSERT_FALSE(ierr);
  ierr = VecSet(P, 1.0);
  ASSERT_FALSE(ierr);
  ierr = VecSetUp(P);
  ASSERT_FALSE(ierr);
  ierr = VecDuplicate(P, &Q);
  ASSERT_FALSE(ierr);
  ierr = VecSetUp(Q);
  ASSERT_FALSE(ierr);

  ierr = A.Action(0.0, P, Q);
  ASSERT_FALSE(ierr);

  ierr = VecSum(Q, &Q_sum);
  ASSERT_FALSE(ierr);
  ASSERT_DOUBLE_EQ(Q_sum, -1.0 * rate_right);

  for (int i_par{0}; i_par < 1; ++i_par) {
    PetscReal dqsum = (i_par == 0) ? -1.0 : 0.0;
    ierr = A.SensAction(i_par, 0.0, P, Q);
    ASSERT_FALSE(ierr);
    ierr = VecSum(Q, &Q_sum);
    ASSERT_FALSE(ierr);
    ASSERT_DOUBLE_EQ(Q_sum, dqsum);
  }

  ierr = VecDestroy(&P);
  ASSERT_FALSE(ierr);
  ierr = VecDestroy(&Q);
  ASSERT_FALSE(ierr);
}

TEST_F(SensMatrixTest, mat_constr_generation) {
  int ierr;

  double Q_sum;

  pacmensl::SensFspMatrix<FspMatrixConstrained> A(PETSC_COMM_WORLD);

  ierr = A.GenerateValues(*state_set, smodel);
  ASSERT_FALSE(ierr);

  Vec P, Q;
  ierr = VecCreate(PETSC_COMM_WORLD, &P);
  ASSERT_FALSE(ierr);
  ierr = VecSetSizes(P, A.GetNumLocalRows(), PETSC_DECIDE);
  ASSERT_FALSE(ierr);
  ierr = VecSetFromOptions(P);
  ASSERT_FALSE(ierr);
  ierr = VecSet(P, 1.0);
  ASSERT_FALSE(ierr);
  ierr = VecSetUp(P);
  ASSERT_FALSE(ierr);
  ierr = VecDuplicate(P, &Q);
  ASSERT_FALSE(ierr);
  ierr = VecSetUp(Q);
  ASSERT_FALSE(ierr);

  ierr = A.Action(0.0, P, Q);
  ASSERT_FALSE(ierr);

  ierr = VecSum(Q, &Q_sum);
  ASSERT_FALSE(ierr);
  ASSERT_DOUBLE_EQ(Q_sum, 0.0);

  for (int i_par{0}; i_par < 2; ++i_par) {
    ierr = A.SensAction(i_par, 0.0, P, Q);
    ASSERT_FALSE(ierr);
    ierr = VecSum(Q, &Q_sum);
    ASSERT_FALSE(ierr);
    ASSERT_DOUBLE_EQ(Q_sum, 0.0);
  }

  ierr = VecDestroy(&P);
  ASSERT_FALSE(ierr);
  ierr = VecDestroy(&Q);
  ASSERT_FALSE(ierr);
}
