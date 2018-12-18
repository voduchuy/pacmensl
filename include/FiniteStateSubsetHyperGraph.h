//
// Created by Huy Vo on 12/15/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H

#include "FiniteStateSubset.h"

namespace cme{
    namespace parallel{
        class FiniteStateSubsetHyperGraph: public FiniteStateSubset{
            struct HyperGraphData{
                int num_local_vertices;
                int num_local_pins;
                ZOLTAN_ID_PTR vtx_gid;
                int *vtx_edge_ptr;
                ZOLTAN_ID_PTR pin_gid;
            } hg_data;

        public:
            PetscLogEvent generate_hg_event, call_zoltan_event, generate_ao_event;
            explicit FiniteStateSubsetHyperGraph(MPI_Comm new_comm);;
            void GenerateStatesAndOrdering() override;
            ~FiniteStateSubsetHyperGraph();

            // Interface to Zoltan
            friend int zoltan_num_obj(void *fss_data, int *ierr);

            friend void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                                        ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                        int *ierr);
            friend void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
            friend void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                    int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);
        };
        int zoltan_num_obj(void *fss_data, int *ierr);

        void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                             int *ierr);
        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                          int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
