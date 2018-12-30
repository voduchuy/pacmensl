//
// Created by Huy Vo on 12/15/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H

#include "FiniteStateSubset.h"

namespace cme {
    namespace parallel {
        class FiniteStateSubsetHyperGraph : public FiniteStateSubset {

        public:
            explicit FiniteStateSubsetHyperGraph(MPI_Comm new_comm);;

            void GenerateStatesAndOrdering() override;

            void ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) override;
        };
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
