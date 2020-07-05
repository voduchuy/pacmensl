//
// Created by Huy Vo on 5/31/19.
//

#ifndef PACMENSL_STATESETPARTITIONER_H
#define PACMENSL_STATESETPARTITIONER_H

#include "StatePartitionerBase.h"
#include "StatePartitionerGraph.h"
#include "StatePartitionerHyperGraph.h"
#include "mpi.h"

// Added something

namespace pacmensl {
class StatePartitioner
{
 private:
  MPI_Comm             comm  = nullptr;
  StatePartitionerBase *data = nullptr;
 public:
  explicit StatePartitioner(MPI_Comm _comm) { comm = _comm; };

  void SetUp(PartitioningType part_type, PartitioningApproach part_approach = PartitioningApproach::REPARTITION);

  int Partition(arma::Mat<int> &states, Zoltan_DD_Struct *state_directory, arma::Mat<int> &stoich_mat,
                int *layout);

  ~StatePartitioner()
  {
    delete data;
    comm = MPI_COMM_NULL;
  }
};
}

#endif //PACMENSL_STATESETPARTITIONER_H
