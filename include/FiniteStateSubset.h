//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSET_H
#define PARALLEL_FSP_FINITESTATESUBSET_H

#include <zoltan.h>
#include "cme_util.h"

namespace cme {
    namespace parallel {
        enum PartitioningType {
            Naive, RCB, Graph, HyperGraph, Hierarch, NotSet
        };

        class FiniteStateSubset {
        protected:
            int set_up = 0;
            int stoich_set = 0;

            MPI_Comm comm;
            PartitioningType partitioning_type;
            arma::Row<PetscInt> fsp_size;
            arma::Mat<PetscInt> stoichiometry;

            PetscInt n_species;
            PetscInt n_states_global;
            PetscInt n_local_states;
            arma::Mat<PetscInt> local_states;

            void LocalStatesFromAO();

            PetscLayout vec_layout = nullptr;
            AO lex2petsc = nullptr;

            arma::Mat<PetscInt> get_my_naive_local_states();

            // This data is needed for partitioning algorithms
            struct AdjacencyData {
                AO lex2zoltan; // Store ordering from FSP states' natural indexing to Zoltan's indexing

                int num_local_states;
                ZOLTAN_ID_PTR states_gid;

                int *num_edges; // Number of states that share information with each local states

                int num_reachable_states; // Number of nz entries on the rows of the FSP matrix corresponding to local states
                ZOLTAN_ID_PTR reachable_states; // Global indices of nz entries on the rows corresponding to local states
                int *reachable_states_proc; // Processors that own the reachable states
                int *edge_ptr;
            } adj_data;

            void GenerateGraphData(arma::Mat<PetscInt> &local_states_tmp);

            void FreeGraphData();

            void GenerateHyperGraphData(arma::Mat<PetscInt> &local_states_tmp);

            void FreeHyperGraphData();

            struct GeomData {
                int dim;
                double *states_coo;
            } geom_data;

            void GenerateGeomData(arma::Row<PetscInt> &fsp_size, arma::Mat<PetscInt> &local_states_tmp);

            void FreeGeomData();

            // These variables are needed for partitioning with Zoltan
            Zoltan_Struct *zoltan;
            // Variables to store Zoltan's output
            int zoltan_err;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            void CallZoltanLoadBalancing();

            void ComputePetscOrderingFromZoltan();

            void FreeZoltanParts();

            /* Zoltan interface functions */
            friend int zoltan_num_obj(void *fss_data, int *ierr);

            friend void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                                        ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                        int *ierr);

            friend int zoltan_num_geom(void *data, int *ierr);

            friend void zoltan_geom_multi(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim,
                                          double *geom_vec, int *ierr);

            friend int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries,
                                        ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);

            friend void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                         ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs,
                                         int wgt_dim, float *ewgts, int *ierr);

            friend void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);

            friend void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                              int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr,
                                              ZOLTAN_ID_PTR pin_gid, int *ierr);

        public:

            // Generic Interface
            explicit FiniteStateSubset(MPI_Comm new_comm);

            MPI_Comm GetComm();

            arma::Row<PetscInt> GetFSPSize();

            void SetStoichiometry(arma::Mat<PetscInt> SM);

            void SetSize(arma::Row<PetscInt> new_fsp_size);

            PetscInt GetNumLocalStates();

            PetscInt GetNumGlobalStates();

            PetscInt GetNumSpecies();

            AO GetAO();

            void PrintAO();

            arma::Mat<PetscInt> GetLocalStates();

            std::tuple<PetscInt, PetscInt> GetLayoutStartEnd();

            arma::Row<PetscInt> State2Petsc(arma::Mat<PetscInt> state);

            void State2Petsc(arma::Mat<PetscInt> state, PetscInt *indx);

            arma::Row<PetscReal> SinkStatesReduce(Vec P);

            void Destroy();

            ~FiniteStateSubset();

            // Implementation-dependent methods
            // This procedure generate data for the members:
            // local_states, vec_layout, lex2petsc
            virtual void GenerateStatesAndOrdering() {};

            virtual void ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) {};

            friend arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);
        };

        arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

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

        /*!
         * Helper functions to convert back and forth between partitioning options and string
         */
        std::string part2str(PartitioningType part);

        PartitioningType str2part(std::string str);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSET_H
