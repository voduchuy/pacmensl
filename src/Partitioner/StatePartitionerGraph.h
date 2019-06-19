//
// Created by Huy Vo on 5/7/19.
//

#ifndef PACMENSL_STATEPARTITIONERGRAPH_H
#define PACMENSL_STATEPARTITIONERGRAPH_H

#include "StatePartitionerBase.h"

namespace pacmensl {
    class StatePartitionerGraph : public StatePartitionerBase {
    protected:
        int *num_edges; ///< Number of states that share information with each local states
        int num_reachable_states; ///< Number of nz entries on the rows of the FSP matrix corresponding to local states
        int *reachable_states; ///< Global indices of nz entries on the rows corresponding to local states
        int *reachable_states_proc; ///< Processors that own the reachable states
        float *edge_weights; ///< For storing the edge weights in graph model
        int *edge_ptr; ///< reachable_states[edge_ptr[i] to ege_ptr[i+1]-1] contains the ids of states connected to local state i

        void set_zoltan_parameters() override;

        void generate_data() override;

        void free_data() override;

        static int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries,
                                    ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);

        static void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                     ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
                                     int *num_edges, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs,
                                     int wgt_dim, float *ewgts, int *ierr);

    public:
        explicit StatePartitionerGraph(MPI_Comm _comm);
    };
}

#endif //PACMENSL_FINITESTATESUBSETGRAPH_H
