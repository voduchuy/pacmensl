//
// Created by Huy Vo on 5/7/19.
//

#ifndef PFSPAT_STATEPARTITIONERHYPERGRAPH_H
#define PFSPAT_STATEPARTITIONERHYPERGRAPH_H

#include "StatePartitionerBase.h"

namespace cme { namespace parallel {
    class StatePartitionerHyperGraph : public StatePartitionerBase {
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

        /* Zoltan interface */
        static void zoltan_hg_size( void *data, int *num_lists, int *num_pins, int *format, int *ierr );
        static void zoltan_hg_edge_list( void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                             ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr );
        static void zoltan_hg_size_eweights( void *data, int *num_edges, int *ierr );
        static void zoltan_hg_edge_weights( void *data, int num_gid_entries, int num_lid_entries, int num_edges,
                                                               int edge_weight_dim, ZOLTAN_ID_PTR edge_GID,
                                                               ZOLTAN_ID_PTR edge_LID, float *edge_weight, int *ierr );
    public:
        StatePartitionerHyperGraph(MPI_Comm _comm): StatePartitionerBase(_comm) {};
        ~StatePartitionerHyperGraph( );
    };
}}
#endif //PFSPAT_FINITESTATESUBSETHYPERGRAPH_H
