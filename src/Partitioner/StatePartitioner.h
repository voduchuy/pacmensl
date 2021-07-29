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
  MPI_Comm             comm = MPI_COMM_NULL;
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
