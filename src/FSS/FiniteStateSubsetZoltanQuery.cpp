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
            *ierr = ZOLTAN_OK;
            return ((FiniteStateSubset*) data)->nstate_local;
        }

        void zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim,
                             float *obj_wgts, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;
            for (int i{0}; i < fss_data->nstate_local; ++i) {
                for (int j{0}; j < num_gid_entries; ++j){
                    global_id[i*num_gid_entries + j] = (ZOLTAN_ID_TYPE ) fss_data->local_states(j, i);
                }
                local_id[i] = (ZOLTAN_ID_TYPE) i;
            }
            if (wgt_dim == 1){
                for (int i{0}; i < fss_data->nstate_local; ++i){
                    obj_wgts[i] = 1;
                }
            }
            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_frontier(void *fss_data, int *ierr){
            *ierr = ZOLTAN_OK;
            return int(((FiniteStateSubset*)fss_data)->frontier_lids.n_elem);
        }


        void zoltan_frontier_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                  ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            int n_frontier = (int) fss_data->frontier_lids.n_elem;
            for (int i{0}; i < n_frontier; ++i) {
                local_ids[i] = (ZOLTAN_ID_TYPE) fss_data->frontier_lids(i);
                for (int j{0}; j < num_gid_entries; ++j){
                    global_id[i*num_gid_entries + j] = (ZOLTAN_ID_TYPE ) fss_data->local_states(j, local_ids[i]);
                }
            }
            if (wgt_dim == 1){
                for (int i{0}; i < n_frontier; ++i){
                    obj_wgts[i] = 1;
                }
            }
            *ierr = ZOLTAN_OK;
        }


//
//        int zoltan_num_geom(void *data, int *ierr) {
//            auto fss_data = (FiniteStateSubset *) data;
//            *ierr = ZOLTAN_OK;
//            return fss_data->geom_data.dim;
//        }
//
//        void
//        zoltan_geom_multi(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids,
//                          ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr) {
//            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
//                *ierr = ZOLTAN_FATAL;
//                return;
//            }
//            auto fss_data = (FiniteStateSubset *) data;
//            ZOLTAN_ID_TYPE local_idx;
//            for (auto i{0}; i < num_obj; ++i) {
//                local_idx = local_ids[i];
//                for (auto j{0}; j < num_dim; ++j) {
//                    geom_vec[num_dim * i + j] = fss_data->geom_data.states_coo[num_dim * local_idx + j];
//                }
//            }
//        }
//
//        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
//            auto *hg_data = (FiniteStateSubset::ConnectivityData *) data;
//            *num_lists = hg_data->num_local_states;
//            *num_pins = hg_data->num_reachable_states;
//            *format = ZOLTAN_COMPRESSED_VERTEX;
//            *ierr = ZOLTAN_OK;
//        }
//
//        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
//                                   ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr) {
//            auto hg_data = (FiniteStateSubset::ConnectivityData *) data;
//
//            if ((num_vertices != hg_data->num_local_states) || (num_pins != hg_data->num_reachable_states) ||
//                (format != ZOLTAN_COMPRESSED_VERTEX)) {
//                *ierr = ZOLTAN_FATAL;
//                return;
//            }
//
//            for (int i{0}; i < num_vertices; ++i) {
//                vtx_gid[i] = (ZOLTAN_ID_TYPE) hg_data->states_gid[i];
//                vtx_edge_ptr[i] = hg_data->edge_ptr[i];
//            }
//
//            for (int i{0}; i < num_pins; ++i) {
//                pin_gid[i] = (ZOLTAN_ID_TYPE) hg_data->reachable_states[i];
//            }
//            *ierr = ZOLTAN_OK;
//        }
//
//        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
//                             ZOLTAN_ID_PTR local_id, int *ierr) {
//            auto g_data = (FiniteStateSubset::ConnectivityData *) data;
//            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
//                *ierr = ZOLTAN_FATAL;
//                return -1;
//            }
//            ZOLTAN_ID_TYPE indx = *local_id;
//            *ierr = ZOLTAN_OK;
//            return g_data->num_edges[indx];
//        }
//
//        void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
//                              ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
//                              float *ewgts, int *ierr) {
//            auto g_data = (FiniteStateSubset::ConnectivityData *) data;
//            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
//                *ierr = ZOLTAN_FATAL;
//                return;
//            }
//            ZOLTAN_ID_TYPE indx = *local_id;
//            int k = 0;
//            int edge_ptr = g_data->edge_ptr[indx];
//            for (auto i = 0; i <  g_data->num_edges[indx]; ++i) {
//                nbor_global_id[k] = (ZOLTAN_ID_TYPE) g_data->reachable_states[edge_ptr + i];
//                k++;
//            }
//            if (wgt_dim == 1){
//                k = 0;
//                for (auto i = 0; i <  g_data->num_edges[indx]; ++i){
//                    ewgts[k] = g_data->edge_weights[edge_ptr + i];
//                    k++;
//                }
//            }
//            *ierr = ZOLTAN_OK;
//        }

        int zoltan_obj_size(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                            ZOLTAN_ID_PTR local_id, int *ierr) {
            *ierr = ZOLTAN_OK;
            auto fss_data = (FiniteStateSubset*) data;
            // The only things that migrate in our current version are the entries of the solution vector
            return (int) sizeof(PetscInt)*
            (
                    fss_data->n_species*(fss_data->n_reactions+1) + fss_data->n_reactions+1
            );
        }

        void
        zoltan_pack_states(void *data, int num_gid_entries, int num_lid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                           ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            if (num_gid_entries != fss_data->n_species || num_lid_entries != 1){
                std::cout << "Error in zoltan_pack_states: number of gid entries does not match CME number of species, or number of lid entries exceed 1.\n";
                *ierr = ZOLTAN_FATAL;
                return;
            }
            for (int i{0}; i < num_ids; ++i){
                auto ptr = (ZOLTAN_ID_TYPE*) &buf[idx[i]];
                auto state_id = local_ids[i];
                // pack the state
                for (int j{0}; j < fss_data->n_species; ++j){
                    *ptr = (ZOLTAN_ID_TYPE) fss_data->local_states(j,state_id);
                    ptr++;
                }
                // pack the state's status
                *ptr = (ZOLTAN_ID_TYPE) fss_data->local_states_status(state_id);
                ptr++;
                // pack the reachable states
                for (int ir{0}; ir < fss_data->n_reactions; ++ir){
                    for (int j{0}; j < fss_data->n_species; ++j){
                        *ptr = (ZOLTAN_ID_TYPE) fss_data->local_reachable_states(j + ir*fss_data->n_species, state_id);
                        ptr++;
                    }
                    *ptr = (ZOLTAN_ID_TYPE) fss_data->local_reachable_states_status(ir, state_id);
                    ptr++;
                }
            }
            // remove the packed states from local data structure
            arma::uvec i_keep = arma::regspace<arma::uvec>(0, 1, fss_data->nstate_local-1);
            for (int i{0}; i < num_ids; ++i){
                arma::uvec idelete = arma::find(i_keep == (arma::uword) local_ids[i]);
                if (!idelete.is_empty()){
                    i_keep.shed_row(idelete[0]);
                }
            }
            fss_data->local_states = fss_data->local_states.cols(i_keep);
            fss_data->local_states_status = fss_data->local_states_status.elem(i_keep).t();
            fss_data->local_reachable_states = fss_data->local_reachable_states.cols(i_keep);
            fss_data->local_reachable_states_status = fss_data->local_reachable_states_status.cols(i_keep);
            fss_data->nstate_local = fss_data->local_states.n_cols;
        }

        void zoltan_unpack_states(void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes,
                                  int *idx, char *buf, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            if (num_gid_entries != fss_data->n_species){
                std::cout << "Error in zoltan_unpack_states: number of gid entries does not match CME number of species.\n";
                *ierr = ZOLTAN_FATAL;
                return;
            }
            int ns = fss_data->n_species;
            int nr = fss_data->n_reactions;
            int nlocal = fss_data->nstate_local;

            // Expand the data arrays
            fss_data->local_states.resize(ns, nlocal+num_ids);
            fss_data->local_states_status.resize(nlocal+num_ids);
            fss_data->local_reachable_states.resize(ns*nr, nlocal+num_ids);
            fss_data->local_reachable_states_status.resize(nr, nlocal+num_ids);

            // Enter data for the arrays
            for (int i{0}; i < num_ids; ++i){
                auto ptr = (ZOLTAN_ID_PTR) &buf[idx[i]];
                for (int j{0}; j < ns; ++j){
                    fss_data->local_states(j, nlocal+i) = *ptr;
                    ptr++;
                }
                fss_data->local_states_status(nlocal+i) = *ptr;
                ptr++;

                for (int ir{0}; ir < nr; ++ir){
                    for (int j{0}; j < ns; ++j){
                        fss_data->local_reachable_states(ir*ns + j, nlocal+i) = *ptr;
                        ptr++;
                    }
                    fss_data->local_reachable_states_status(ir, nlocal+i) = *ptr;
                    ptr++;
                }
            }
        }
    }
}
