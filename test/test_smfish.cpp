//
// Created by Huy Vo on 2019-06-09.
//
static char help[] = "Test smFish data object.\n\n";

#include <pacmensl_all.h>

using namespace pacmensl;

int main(int argc, char *argv[]) {
  //PACMENSL parallel environment object, must be created before using other PACMENSL's functionalities
  pacmensl::Environment my_env(&argc, &argv, help);

  bool pass = true;

  MPI_Comm comm;
  int num_proc, rank;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &num_proc);
  MPI_Comm_rank(comm, &rank);

  DiscreteDistribution distribution;
  distribution.comm_ = comm;
  distribution.states.set_size(1, 10);
  for (int i = 0; i < 10; ++i) {
    distribution.states(0, i) = i;
  }

  VecCreate(comm, &distribution.p);
  VecSetType(distribution.p, VECMPI);
  VecSetSizes(distribution.p, 10, PETSC_DECIDE);
  VecSet(distribution.p, 0.1 / double(num_proc));
  VecSetUp(distribution.p);
  VecAssemblyBegin(distribution.p);
  VecAssemblyEnd(distribution.p);

  arma::Row<int> freq(5, arma::fill::ones);
  SmFishSnapshot data(distribution.states.cols(arma::span(0, 4)), freq);

  double ll = SmFishSnapshotLogLikelihood(data, distribution);
  double ll_true = 5.0 * log(0.1);
  if (abs(ll - ll_true) > 1.0e-15) pass = false;

  if (!pass) return -1;
  return 0;
}