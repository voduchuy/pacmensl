//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETPARMETIS_H
#define PARALLEL_FSP_FINITESTATESUBSETPARMETIS_H

#include "FiniteStateSubset.h"

namespace cme{
    namespace petsc{
        class FiniteStateSubsetParMetis: public FiniteStateSubset{
        public:
            explicit FiniteStateSubsetParMetis(MPI_Comm new_comm): FiniteStateSubset(new_comm) {partitioning_type = ParMetis;};
            void GenerateStatesAndOrdering() override;
        };
    }
}



#endif //PARALLEL_FSP_FINITESTATESUBSETPARMETIS_H
