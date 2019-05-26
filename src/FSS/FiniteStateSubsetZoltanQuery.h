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

        void zoltan_get_hg_size_eweights(void *data, int *num_edges, int *ierr);

        void zoltan_get_hg_eweights(void *data, int num_gid_entries, int num_lid_entries, int num_edges, int edge_weight_dim, ZOLTAN_ID_PTR edge_GID, ZOLTAN_ID_PTR edge_LID, float  *edge_weight, int *ierr);

        int zoltan_hier_num_levels (void *data, int *ierr);

        int zoltan_hier_part (void *data, int level, int *ierr);

        void zoltan_hier_method (void *data, int level, struct Zoltan_Struct * zz, int *ierr);

        /* Zoltan migration functions */

        /// Zoltan interface to give Zoltan the sizes of local states data
        int zoltan_obj_size(
                void *data,
                int num_gid_entries,
                int num_lid_entries,
                ZOLTAN_ID_PTR global_id,
                ZOLTAN_ID_PTR local_id,
                int *ierr);

        /// Zoltan interface to pack frontiers for migrating
        void zoltan_pack_frontiers (
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

        /// Zoltan interface for processing local data structure mid-migration
        void zoltan_frontiers_mid_migrate_pp(
                void *data,
                int num_gid_entries,
                int num_lid_entries,
                int num_import,
                ZOLTAN_ID_PTR import_global_ids,
                ZOLTAN_ID_PTR import_local_ids,
                int *import_procs,
                int *import_to_part,
                int num_export,
                ZOLTAN_ID_PTR export_global_ids,
                ZOLTAN_ID_PTR export_local_ids,
                int *export_procs,
                int *export_to_part,
                int *ierr);

        /// Zoltan interface to unpack states migrated from other processors
        void zoltan_unpack_frontiers (
                void *data,
                int num_gid_entries,
                int num_ids,
                ZOLTAN_ID_PTR global_ids,
                int *sizes,
                int *idx,
                char *buf,
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

        /// Zoltan interface for processing local data structure mid-migration
        void zoltan_mid_migrate_pp(
                void *data,
                int num_gid_entries,
                int num_lid_entries,
                int num_import,
                ZOLTAN_ID_PTR import_global_ids,
                ZOLTAN_ID_PTR import_local_ids,
                int *import_procs,
                int *import_to_part,
                int num_export,
                ZOLTAN_ID_PTR export_global_ids,
                ZOLTAN_ID_PTR export_local_ids,
                int *export_procs,
                int *export_to_part,
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
