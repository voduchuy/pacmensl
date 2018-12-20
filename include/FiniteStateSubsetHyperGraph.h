//
// Created by Huy Vo on 12/15/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H

#include "FiniteStateSubset.h"

namespace cme {
    namespace parallel {
        class FiniteStateSubsetHyperGraph : public FiniteStateSubset {
            void generate_hypergraph_data(arma::Mat<PetscInt> &local_states);

            void free_hypergraph_data();
        public:
            PetscLogEvent generate_hg_event, call_zoltan_event, generate_ao_event;

            explicit FiniteStateSubsetHyperGraph(MPI_Comm new_comm);;

            void GenerateStatesAndOrdering() override;

            void ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) override;

            ~FiniteStateSubsetHyperGraph();
        };
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
