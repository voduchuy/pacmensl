//
// Created by Huy Vo on 12/3/18.
//
#include<gtest/gtest.h>
#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"Sys.h"
#include"StateSetConstrained.h"
#include"pacmensl_test_env.h"

static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

using namespace pacmensl;

// Test if StateSet can detect wrong inputs into constructors and setters and throws the appropriate exceptions
TEST(StateSetExpansion, toggle_state_set_insertion_error_handling) {
  // Stoichiometri matrix
  arma::Mat<int> SM{{1, -1, 0, 0},
                    {0, 0, 1, -1}};

  arma::Mat<int> X0(3, 1);
  X0.col(0).fill(0);

  // Constraint function
  fsp_constr_multi_fn constr_fun = [&](int n_species, int n_constraints, int n_states, int *states, int *output,
                                       void *args) {
    if (n_constraints != 1) {
      return int(-1);
    }
    if (n_species != 2) {
      return int(-1);
    }
    for (int i{0}; i < n_states; ++i) {
      output[i] = states[2 * i] + states[2 * i + 1];
    }
    return int(0);
  };

  StateSetConstrained state_set(MPI_COMM_WORLD);
  ASSERT_EQ(state_set.SetStoichiometryMatrix(SM), 0);
  ASSERT_EQ(state_set.AddStates(X0), -1);
}




TEST(StateSetExpansion, toggle_state_set_expansion_lb_naive) {
  int      ierr;
  MPI_Comm comm;
  ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  ASSERT_FALSE(ierr);
  int my_rank;
  ierr = MPI_Comm_rank(comm, &my_rank);
  ASSERT_FALSE(ierr);

  // Stoichiometri matrix
  arma::Mat<int> SM{{1, -1, 0, 0},
                    {0, 0, 1, -1}};

  arma::Mat<int> X0(2, 1);
  X0.col(0).fill(0);

  // Constraint function
  fsp_constr_multi_fn constr_fun = [&](int n_species, int n_constraints, int n_states, int *states, int *output,
                                       void *args) {
    if (n_constraints != 1) {
      return int(-1);
    }
    if (n_species != 2) {
      return int(-1);
    }
    for (int i{0}; i < n_states; ++i) {
      output[i] = states[2 * i] + states[2 * i + 1];
    }
    return int(0);
  };

  StateSetConstrained state_set(comm);

  // Generate a small Fsp
  arma::Row<int> fsp_size = {3};
  ASSERT_EQ(state_set.SetStoichiometryMatrix(SM), 0);
  ASSERT_EQ(state_set.SetLoadBalancingScheme(PartitioningType::BLOCK), 0);
  ASSERT_EQ(state_set.SetShape(constr_fun, fsp_size), 0);
  ASSERT_EQ(state_set.SetUp(), 0);
  ASSERT_EQ(state_set.AddStates(X0), 0);
  ASSERT_EQ(state_set.Expand(), 0);

  // Make sure the size is correct
  int num_states_global = state_set.GetNumGlobalStates();
  ASSERT_EQ(num_states_global, 10);

  // Make sure the content is correct
  int      c{0};
  int      all_states[2 * 10];
  int      indx[10];
  for (int i{0}; i < 4; ++i) {
    for (int j{0}; j < 4 - i; ++j) {
      all_states[2 * c]     = i;
      all_states[2 * c + 1] = j;
      c++;
    }
  }
  state_set.State2Index(10, &all_states[0], &indx[0]);
  for (int i{0}; i < 10; ++i) {
    ASSERT_GE(indx[i], 0);
  }

  ierr = MPI_Comm_free(&comm);
  ASSERT_FALSE(ierr);
}

TEST(StateSetExpansion, toggle_state_set_expansion_lb_graph) {
  int      ierr;
  MPI_Comm comm;
  ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  ASSERT_FALSE(ierr);
  int my_rank;
  ierr = MPI_Comm_rank(comm, &my_rank);
  ASSERT_FALSE(ierr);

  // Stoichiometri matrix
  arma::Mat<int> SM{{1, -1, 0, 0},
                    {0, 0, 1, -1}};

  arma::Mat<int> X0(2, 1);
  X0.col(0).fill(0);

  // Constraint function
  fsp_constr_multi_fn constr_fun = [&](int n_species, int n_constraints, int n_states, int *states, int *output,
                                       void *args) {
    if (n_constraints != 1) {
      return int(-1);
    }
    if (n_species != 2) {
      return int(-1);
    }
    for (int i{0}; i < n_states; ++i) {
      output[i] = states[2 * i] + states[2 * i + 1];
    }
    return int(0);
  };

  StateSetConstrained state_set(comm);
  state_set.SetStoichiometryMatrix(SM);

  // Generate a small Fsp
  arma::Row<int> fsp_size = {3};
  state_set.SetShape(constr_fun, fsp_size);
  state_set.SetLoadBalancingScheme(PartitioningType::GRAPH);
  state_set.SetUp();
  state_set.AddStates(X0);
  state_set.Expand();

  // Make sure the size is correct
  int num_states_global = state_set.GetNumGlobalStates();
  ASSERT_EQ(num_states_global, 10);

  // Make sure the content is correct
  int      c{0};
  int      all_states[2 * 10];
  int      indx[10];
  for (int i{0}; i < 4; ++i) {
    for (int j{0}; j < 4 - i; ++j) {
      all_states[2 * c]     = i;
      all_states[2 * c + 1] = j;
      c++;
    }
  }
  state_set.State2Index(10, &all_states[0], &indx[0]);
  for (int i{0}; i < 10; ++i) {
    ASSERT_GE(indx[i], 0);
  }

  ierr = MPI_Comm_free(&comm);
  ASSERT_FALSE(ierr);
}

TEST(StateSetExpansion, toggle_state_set_expansion_lb_hypergraph) {
  int      ierr;
  MPI_Comm comm;
  ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  ASSERT_FALSE(ierr);
  int my_rank;
  ierr = MPI_Comm_rank(comm, &my_rank);
  ASSERT_FALSE(ierr);

  // Stoichiometri matrix
  arma::Mat<int> SM{{1, -1, 0, 0},
                    {0, 0, 1, -1}};

  arma::Mat<int> X0(2, 1);
  X0.col(0).fill(0);

  // Constraint function
  fsp_constr_multi_fn constr_fun = [&](int n_species, int n_constraints, int n_states, int *states, int *output,
                                       void *args) {
    if (n_constraints != 1) {
      return int(-1);
    }
    if (n_species != 2) {
      return int(-1);
    }
    for (int i{0}; i < n_states; ++i) {
      output[i] = states[2 * i] + states[2 * i + 1];
    }
    return int(0);
  };

  StateSetConstrained state_set(comm);
  state_set.SetStoichiometryMatrix(SM);

  // Generate a small Fsp
  arma::Row<int> fsp_size = {3};
  state_set.SetLoadBalancingScheme(PartitioningType::HYPERGRAPH);
  state_set.SetShape(constr_fun, fsp_size);
  state_set.SetUp();
  state_set.AddStates(X0);
  state_set.Expand();

  // Make sure the size is correct
  int num_states_global = state_set.GetNumGlobalStates();
  ASSERT_EQ(num_states_global, 10);

  // Make sure the content is correct
  int      c{0};
  int      all_states[2 * 10];
  int      indx[10];
  for (int i{0}; i < 4; ++i) {
    for (int j{0}; j < 4 - i; ++j) {
      all_states[2 * c]     = i;
      all_states[2 * c + 1] = j;
      c++;
    }
  }
  state_set.State2Index(10, &all_states[0], &indx[0]);
  for (int i{0}; i < 10; ++i) {
    ASSERT_GE(indx[i], 0);
  }

  ierr = MPI_Comm_free(&comm);
  ASSERT_FALSE(ierr);
}
