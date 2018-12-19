//
// Created by Huy Vo on 12/15/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H

#include "FiniteStateSubset.h"

namespace cme{
    namespace parallel{
        class FiniteStateSubsetHyperGraph: public FiniteStateSubset{

        public:
            PetscLogEvent generate_hg_event, call_zoltan_event, generate_ao_event;
            explicit FiniteStateSubsetHyperGraph(MPI_Comm new_comm);;
            void GenerateStatesAndOrdering() override;
            ~FiniteStateSubsetHyperGraph();
        };
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
