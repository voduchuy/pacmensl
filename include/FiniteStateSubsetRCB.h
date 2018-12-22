//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETRCB_H
#define PARALLEL_FSP_FINITESTATESUBSETRCB_H

#include "FiniteStateSubset.h"

namespace cme {
    namespace parallel {
        class FiniteStateSubsetRCB : public FiniteStateSubset {
        public:
            explicit FiniteStateSubsetRCB(MPI_Comm new_comm);;

            void GenerateStatesAndOrdering() override;

            void ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) override;
        };
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETRCB_H
