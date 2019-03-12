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
                    obj_wgts[i] = fss_data->state_weights(i);
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

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
            auto *hg_data = (FiniteStateSubset *) data;
            *num_lists = hg_data->nstate_local;
            *num_pins = (int) arma::sum(hg_data->num_local_edges);
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                   ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr) {
            auto hg_data = (FiniteStateSubset*) data;

            int ns {hg_data->n_species}, nr {hg_data->n_reactions};
            if ((num_vertices != hg_data->nstate_local) ||
                (format != ZOLTAN_COMPRESSED_VERTEX)) {
                *ierr = ZOLTAN_FATAL;
                std::cout << "zoltan_get_hypergraph fails. \n";
                return;
            }

            int pin_ptr{0};
            for (int i{0}; i < num_vertices; ++i) {
                if (i == 0){
                    vtx_edge_ptr[0] = 0;
                }
                else{
                    vtx_edge_ptr[i] = vtx_edge_ptr[i-1] + hg_data->num_local_edges(i-1);
                }
                if (pin_ptr != vtx_edge_ptr[i]*ns){
                    *ierr = ZOLTAN_FATAL;
                    return;
                }
                for (int ii{0}; ii < ns; ++ii){
                    vtx_gid[i*ns + ii] = (ZOLTAN_ID_TYPE) hg_data->local_states(ii, i);
                }
                for (int ir{0}; ir < nr; ++ir){
                    if (hg_data->local_observable_states_status(ir, i) == 0){
                        for (int ii{0}; ii < ns; ++ii){
                            pin_gid[pin_ptr + ii] = (ZOLTAN_ID_TYPE) hg_data->local_observable_states(ir*ns + ii, i);
                        }
                        pin_ptr += ns;
                    }
                }
            }
            vtx_edge_ptr[num_vertices] = arma::sum(hg_data->num_local_edges);
            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                             ZOLTAN_ID_PTR local_id, int *ierr) {
            auto g_data = (FiniteStateSubset*) data;
            if ((num_gid_entries != g_data->n_species) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return -1;
            }
            ZOLTAN_ID_TYPE indx = *local_id;
            *ierr = ZOLTAN_OK;
            return g_data->num_local_edges[indx];
        }

        void zoltan_get_graph_edges(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
                int *num_edges, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim, float *ewgts, int *ierr){
            auto g_data = (FiniteStateSubset*) data;
            if ((num_gid_entries != g_data->n_species) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            int n = g_data->nstate_local;
            int ns = g_data->n_species;
            int nr = g_data->n_reactions;
            int nbr_ptr = 0;
            for (int i{0}; i < n; ++i){
                for (int ii{0}; ii < ns; ++ii){
                    global_id[i*ns + ii] = (ZOLTAN_ID_TYPE) g_data->local_states(ii, i);
                }
                local_id[i] = (ZOLTAN_ID_TYPE) i;
                for (int ir{0}; ir < nr; ++ir){
                    if (g_data->local_observable_states_status(ir, i) == 0){
                        for (int ii{0}; ii < ns; ++ii){
                            nbor_global_id[nbr_ptr + ii] = (ZOLTAN_ID_TYPE) g_data->local_observable_states(ir*ns + ii, i);
                        }
                        nbr_ptr += ns;
                    }
                }
            }
            *ierr = ZOLTAN_OK;
        }

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
