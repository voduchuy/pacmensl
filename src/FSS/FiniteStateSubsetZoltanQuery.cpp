//
// Created by Huy Vo on 12/4/18.
//
#include <FSS/FiniteStateSubset.h>
#include <FSS/FiniteStateSubsetZoltanQuery.h>
#include "FiniteStateSubset.h"
#include "FiniteStateSubsetZoltanQuery.h"


namespace cme {
    namespace parallel {

        // Interface to Zoltan
        int zoltan_num_obj(void *data, int *ierr) {
            *ierr = 0;
            return ((FiniteStateSubset::ConnectivityData *) data)->num_local_states;
        }

        void zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim,
                             float *obj_wgts, int *ierr) {
            auto adj_data = (FiniteStateSubset::ConnectivityData *) data;
            for (int i{0}; i < adj_data->num_local_states; ++i) {
                global_id[i] = (ZOLTAN_ID_TYPE ) adj_data->states_gid[i];
                local_id[i] = (ZOLTAN_ID_TYPE) i;
            }
            if (wgt_dim == 1){
                for (int i{0}; i < adj_data->num_local_states; ++i){
                    obj_wgts[i] = adj_data->states_weights[i];
                }
            }
            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_geom(void *data, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;
            *ierr = ZOLTAN_OK;
            return fss_data->geom_data.dim;
        }

        void
        zoltan_geom_multi(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids,
                          ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr) {
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            auto fss_data = (FiniteStateSubset *) data;
            ZOLTAN_ID_TYPE local_idx;
            for (auto i{0}; i < num_obj; ++i) {
                local_idx = local_ids[i];
                for (auto j{0}; j < num_dim; ++j) {
                    geom_vec[num_dim * i + j] = fss_data->geom_data.states_coo[num_dim * local_idx + j];
                }
            }
        }

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
            auto *hg_data = (FiniteStateSubset::ConnectivityData *) data;
            *num_lists = hg_data->num_local_states;
            *num_pins = hg_data->num_reachable_states;
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                   ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr) {
            auto hg_data = (FiniteStateSubset::ConnectivityData *) data;

            if ((num_vertices != hg_data->num_local_states) || (num_pins != hg_data->num_reachable_states) ||
                (format != ZOLTAN_COMPRESSED_VERTEX)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }

            for (int i{0}; i < num_vertices; ++i) {
                vtx_gid[i] = (ZOLTAN_ID_TYPE) hg_data->states_gid[i];
                vtx_edge_ptr[i] = hg_data->edge_ptr[i];
            }

            for (int i{0}; i < num_pins; ++i) {
                pin_gid[i] = (ZOLTAN_ID_TYPE) hg_data->reachable_states[i];
            }
            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                             ZOLTAN_ID_PTR local_id, int *ierr) {
            auto g_data = (FiniteStateSubset::ConnectivityData *) data;
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return -1;
            }
            ZOLTAN_ID_TYPE indx = *local_id;
            *ierr = ZOLTAN_OK;
            return g_data->num_edges[indx];
        }

        void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                              ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                              float *ewgts, int *ierr) {
            auto g_data = (FiniteStateSubset::ConnectivityData *) data;
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            ZOLTAN_ID_TYPE indx = *local_id;
            int k = 0;
            int edge_ptr = g_data->edge_ptr[indx];
            for (auto i = 0; i <  g_data->num_edges[indx]; ++i) {
                nbor_global_id[k] = (ZOLTAN_ID_TYPE) g_data->reachable_states[edge_ptr + i];
                k++;
            }
            if (wgt_dim == 1){
                k = 0;
                for (auto i = 0; i <  g_data->num_edges[indx]; ++i){
                    ewgts[k] = g_data->edge_weights[edge_ptr + i];
                    k++;
                }
            }
            *ierr = ZOLTAN_OK;
        }

        int zoltan_obj_size(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                            ZOLTAN_ID_PTR local_id, int *ierr) {
            *ierr = ZOLTAN_OK;
            // The only things that migrate in our current version are the entries of the solution vector
            return (int) sizeof(PetscReal);
        }
    }
}
