//
// Created by Huy Vo on 2019-06-09.
//
static char help[] = "Test smFish data object.\n\n";

#include <gtest/gtest.h>
#include <pacmensl_all.h>
#include "my_test_env.h"

using namespace pacmensl;

TEST(SmFISH, log_likelihood) {
  MPI_Comm comm;
  int num_proc, rank;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &num_proc);
  MPI_Comm_rank(comm, &rank);

  DiscreteDistribution distribution;
  distribution.comm_ = comm;
  distribution.states_.set_size(1, 10);
  for (int i = 0; i < 10; ++i) {
    distribution.states_(0, i) = i;
  }

  VecCreate(comm, &distribution.p_);
  VecSetType(distribution.p_, VECMPI);
  VecSetSizes(distribution.p_, 10, PETSC_DECIDE);
  VecSet(distribution.p_, 0.1 / double(num_proc));
  VecSetUp(distribution.p_);
  VecAssemblyBegin(distribution.p_);
  VecAssemblyEnd(distribution.p_);

  arma::Row<int> freq(5, arma::fill::ones);
  SmFishSnapshot data(distribution.states_.cols(arma::span(0, 4)), freq);

  double ll = SmFishSnapshotLogLikelihood(data, distribution);
  double ll_true = 5.0 * log(0.1);
  ASSERT_LE(abs(ll - ll_true), 1.0e-15);
}