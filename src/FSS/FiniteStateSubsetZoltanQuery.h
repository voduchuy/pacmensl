//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTANQUERY_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTANQUERY_H

#include <zoltan.h>
#include "util/cme_util.h"

namespace cme {
    namespace parallel {

        /* Zoltan interface functions */
        int zoltan_num_obj(void *fss_data, int *ierr);

        void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                             int *ierr);

        int zoltan_num_geom(void *data, int *ierr);

        void zoltan_geom_multi(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                               ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec,
                               int *ierr);

        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);

        void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                              ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                              float *ewgts, int *ierr);

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                   int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid,
                                   int *ierr);

        int zoltan_obj_size(
                void *data,
                int num_gid_entries,
                int num_lid_entries,
                ZOLTAN_ID_PTR global_id,
                ZOLTAN_ID_PTR local_id,
                int *ierr);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTANQUERY_H
