//
// Created by Huy Vo on 5/31/19.
//

#ifndef PFSPAT_STATESETPARTITIONER_H
#define PFSPAT_STATESETPARTITIONER_H

#include "StatePartitionerBase.h"
#include "StatePartitionerGraph.h"
#include "StatePartitionerHyperGraph.h"
#include "mpi.h"

// Added something

namespace cme {
    namespace parallel {
        class StatePartitioner {
        private:
            MPI_Comm comm = nullptr;
            StatePartitionerBase *base = nullptr;
        public:
            explicit StatePartitioner( MPI_Comm _comm ) { MPI_Comm_dup( _comm, &comm ); };
            void set_up(PartitioningType part_type, PartitioningApproach part_approach = Repartition);
            void partition( arma::Mat< int > &states, Zoltan_DD_Struct *state_directory, arma::Mat< int > &stoich_mat,
                                        int *layout );
            ~StatePartitioner( ) {
                MPI_Comm_free( &comm );
                delete base;
            }
        };
    }
}

#endif //PFSPAT_STATESETPARTITIONER_H
