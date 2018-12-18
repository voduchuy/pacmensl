//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETLINEAR_H
#define PARALLEL_FSP_FINITESTATESUBSETLINEAR_H

#include "FiniteStateSubset.h"

namespace cme{
    namespace parallel{
        class FiniteStateSubsetNaive: public FiniteStateSubset{
        public:
            explicit FiniteStateSubsetNaive(MPI_Comm new_comm): FiniteStateSubset(new_comm) { partitioning_type = Naive;};

            void GenerateStatesAndOrdering() override;
        };
    }
}

#endif //PARALLEL_FSP_FINITESTATESUBSETLINEAR_H
