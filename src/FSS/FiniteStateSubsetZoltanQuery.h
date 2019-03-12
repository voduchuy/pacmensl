//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTANQUERY_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTANQUERY_H

#include <zoltan.h>
#include "util/cme_util.h"

namespace cme {
    namespace parallel {

        /* Zoltan load-balancing functions */
        int zoltan_num_obj(void *fss_data, int *ierr);

        void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                             int *ierr);

        int zoltan_num_frontier(void *fss_data, int *ierr);

        void zoltan_frontier_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                                  ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                  int *ierr);

        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);

        void zoltan_get_graph_edges(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
                                    int *num_edges, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim, float *ewgts, int *ierr);

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                   int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid,
                                   int *ierr);

        /* Zoltan migration functions */

        /// Zoltan interface to give Zoltan the sizes of local states data
        int zoltan_obj_size(
                void *data,
                int num_gid_entries,
                int num_lid_entries,
                ZOLTAN_ID_PTR global_id,
                ZOLTAN_ID_PTR local_id,
                int *ierr);

        /// Zoltan interface to pack states for migrating
        void zoltan_pack_states (
                void *data,
                int num_gid_entries,
                int num_lid_entries,
                int num_ids,
                ZOLTAN_ID_PTR global_ids,
                ZOLTAN_ID_PTR local_ids,
                int *dest,
                int *sizes,
                int *idx,
                char *buf,
                int *ierr);

        /// Zoltan interface to unpack states migrated from other processors
        void zoltan_unpack_states (
                void *data,
                int num_gid_entries,
                int num_ids,
                ZOLTAN_ID_PTR global_ids,
                int *sizes,
                int *idx,
                char *buf,
                int *ierr);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTANQUERY_H
