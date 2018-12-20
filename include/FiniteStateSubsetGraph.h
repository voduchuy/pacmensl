//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETPARMETIS_H
#define PARALLEL_FSP_FINITESTATESUBSETPARMETIS_H

#include "FiniteStateSubset.h"

namespace cme {
    namespace parallel {
        class FiniteStateSubsetGraph : public FiniteStateSubset {
            inline void generate_graph_data(arma::Mat<PetscInt>& local_states_tmp);
            inline void free_graph_data();
        public:
            explicit FiniteStateSubsetGraph(MPI_Comm new_comm);;

            void GenerateStatesAndOrdering() override;

            void ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) override;
        };
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETPARMETIS_H
