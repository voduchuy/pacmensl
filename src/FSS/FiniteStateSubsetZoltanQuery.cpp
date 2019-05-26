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
            return ((FiniteStateSubset *) data)->GetNumLocalStates();
        }

        void zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim,
                             float *obj_wgts, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;

            fss_data->GiveZoltanObjList(num_gid_entries, num_lid_entries, global_id, local_id, wgt_dim, obj_wgts, ierr);

            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_frontier(void *fss_data, int *ierr) {
            *ierr = ZOLTAN_OK;
            return ((FiniteStateSubset *) fss_data)->GiveZoltanNumFrontier();
        }


        void zoltan_frontier_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                  ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;
            fss_data->GiveZoltanFrontierList(num_gid_entries, num_lid_entries, global_id, local_ids, wgt_dim, obj_wgts, ierr);
        }

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
            auto *hg_data = (FiniteStateSubset *) data;
            hg_data->GiveZoltanHypergraphSize(num_lists, num_pins, format, ierr);
        }

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                   ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr) {
            auto hg_data = (FiniteStateSubset *) data;

            if ((num_vertices != hg_data->GetNumLocalStates()) ||
                (format != ZOLTAN_COMPRESSED_VERTEX)) {
                *ierr = ZOLTAN_FATAL;
                std::cout << "zoltan_get_hypergraph fails. \n";
                return;
            }

            hg_data->GiveZoltanHypergraph(num_gid_entries, num_vertices, num_pins, format, vtx_gid, vtx_edge_ptr, pin_gid, ierr);
        }

        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                             ZOLTAN_ID_PTR local_id, int *ierr) {
            auto g_data = (FiniteStateSubset *) data;
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return -1;
            }
            return g_data->GiveZoltanNumEdges(num_gid_entries, num_lid_entries, global_id, local_id, ierr);
        }

        void zoltan_get_graph_edges(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                    ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
                                    int *num_edges, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                                    float *ewgts, int *ierr) {
            auto g_data = (FiniteStateSubset *) data;
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            g_data->GiveZoltanGraphEdges(num_gid_entries,num_lid_entries, num_obj, global_id, local_id, num_edges, nbor_global_id, nbor_procs, wgt_dim, ewgts, ierr);
        }

        void zoltan_get_hg_size_eweights( void *data, int *num_edges, int *ierr ) {
            auto fss = (FiniteStateSubset*) data;
            *num_edges = fss->GetNumLocalStates();
            *ierr = ZOLTAN_OK;
        }

        void zoltan_get_hg_eweights (void *data, int num_gid_entries, int num_lid_entries, int num_edges, int edge_weight_dim, ZOLTAN_ID_PTR edge_GID, ZOLTAN_ID_PTR edge_LID, float  *edge_weight, int *ierr){
            auto fss = (FiniteStateSubset*) data;
            fss->GiveZoltanHypergraphEdgeWeights(num_gid_entries, num_lid_entries, num_edges, edge_weight_dim, edge_GID, edge_LID, edge_weight, ierr);
            *ierr = ZOLTAN_OK;
        }

        int zoltan_obj_size(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                            ZOLTAN_ID_PTR local_id, int *ierr) {
            *ierr = ZOLTAN_OK;
            auto fss_data = (FiniteStateSubset *) data;
            // The only things that migrate in our current version are the entries of the solution vector
            return fss_data->GiveZoltanObjSize(num_gid_entries, num_lid_entries, global_id, local_id, ierr);
        }

        void zoltan_pack_frontiers( void *data, int num_gid_entries, int num_lid_entries, int num_ids,
                                    ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx,
                                    char *buf, int *ierr ) {
            auto fss_data = (FiniteStateSubset *) data;
            if (num_gid_entries != 1 || num_lid_entries != 1) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            fss_data->PackFrontiers(num_gid_entries, num_lid_entries, num_ids, global_ids, local_ids, dest, sizes, idx, buf, ierr);
        }

        void zoltan_frontiers_mid_migrate_pp( void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                              ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                              int *import_procs, int *import_to_part, int num_export,
                                              ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
                                              int *export_procs, int *export_to_part, int *ierr ) {
            auto fss_data = (FiniteStateSubset *) data;
            if (num_gid_entries != 1) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            fss_data->FrontiersMidMigration(num_gid_entries, num_lid_entries, num_import, nullptr, nullptr, nullptr, nullptr, num_export, export_global_ids, export_local_ids, nullptr, nullptr,nullptr);
            *ierr = ZOLTAN_OK;
        }

        void
        zoltan_unpack_frontiers( void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes,
                                 int *idx, char *buf, int *ierr ) {
            auto fss_data = (FiniteStateSubset *) data;
            if (num_gid_entries != 1) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            fss_data->ReceiveFrontiers(num_gid_entries, num_ids, global_ids, sizes, idx, buf, ierr);
        }

        void
        zoltan_pack_states(void *data, int num_gid_entries, int num_lid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                           ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;
            if (num_gid_entries != 1 || num_lid_entries != 1) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            fss_data->GiveZoltanSendBuffer(num_gid_entries, num_lid_entries, num_ids, global_ids, local_ids, dest, sizes, idx, buf, ierr);
        }

        void zoltan_unpack_states(void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes,
                                  int *idx, char *buf, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;
            if (num_gid_entries != 1) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            fss_data->ReceiveZoltanBuffer(num_gid_entries, num_ids, global_ids, sizes, idx, buf, ierr);
        }


        void zoltan_mid_migrate_pp(void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                   ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids, int *import_procs,
                                   int *import_to_part, int num_export, ZOLTAN_ID_PTR export_global_ids,
                                   ZOLTAN_ID_PTR export_local_ids, int *export_procs, int *export_to_part, int *ierr) {
            auto fss_data = (FiniteStateSubset *) data;
            if (num_gid_entries != 1) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            fss_data->MidMigrationProcessing(num_gid_entries, num_lid_entries, num_import, nullptr, nullptr, nullptr, nullptr, num_export, export_global_ids, export_local_ids, nullptr, nullptr,nullptr);
            *ierr = ZOLTAN_OK;
        }

        int zoltan_hier_num_levels(void *data, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            *ierr = ZOLTAN_OK;
            return fss_data->num_levels;
        }

        int zoltan_hier_part(void *data, int level, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            if (level >= fss_data->num_levels) {
                *ierr = ZOLTAN_FATAL;
                std::cout
                        << "Zoltan_hier_part requests higher level than the max number of levels in the hierarchical partitioning.\n";
                return -1;
            }
            *ierr = ZOLTAN_OK;
            return ((int) fss_data->my_part[level]);
        }

        void zoltan_hier_method(void *data, int level, struct Zoltan_Struct *zz, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            if (level >= fss_data->num_levels) {
                *ierr = ZOLTAN_FATAL;
                std::cout
                        << "Zoltan_hier_part requests higher level than the max number of levels in the hierarchical partitioning.\n";
                return;
            }
            fss_data->SetHiearchicalMethods(level, zz);
            *ierr = ZOLTAN_OK;
        }

    }
}
