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
    class StatePartitioner {
    private:
        MPI_Comm comm = nullptr;
        StatePartitionerBase *base = nullptr;
    public:
        explicit StatePartitioner(MPI_Comm _comm) { MPI_Comm_dup(_comm, &comm); };

        void set_up(PartitioningType part_type, PartitioningApproach part_approach = REPARTITION);

        void partition(arma::Mat<int> &states, Zoltan_DD_Struct *state_directory, arma::Mat<int> &stoich_mat,
                       int *layout);

        ~StatePartitioner() {
            MPI_Comm_free(&comm);
            delete base;
        }
    };
}


#endif //PACMENSL_STATESETPARTITIONER_H
