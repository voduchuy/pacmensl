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
static char help[] = "Test smFish data object.\n\n";

#include <gtest/gtest.h>
#include <pacmensl_all.h>
#include "pacmensl_test_env.h"

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