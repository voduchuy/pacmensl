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
            Naive, Graph, HyperGraph, NotSet
        };

        class FiniteStateSubset {
        protected:
            // This data is needed for partitioning algorithms
            struct AdjacencyData{
                int num_local_states;
                int num_reachable_states_rows; // Number of nz entries on the rows of the FSP matrix corresponding to local states
                int num_reachable_states_cols; // Number of nz entries on the columns of the FSP matrix corresponding to local states (only needed when using Graph partitioning)
                ZOLTAN_ID_PTR states_gid;
                ZOLTAN_ID_PTR reachable_states_rows_gid; // Global indices of nz entries on the rows corresponding to local states
                ZOLTAN_ID_PTR reachable_states_cols_gid; // Global indices of nz entries on the columns corresponding to local states
                int *reachable_states_rows_proc; // Processors that own the reachable states
                int *reachable_states_cols_proc; // Processors that own the reachable states
                int *rows_edge_ptr;
                int *cols_edge_ptr;
                int *num_edges; // Number of states that share information with each local states
            } adj_data;
            int set_up = 0;
            int stoich_set = 0;
            MPI_Comm comm;

            PartitioningType partitioning_type;
            arma::Row<PetscInt> fsp_size;
            PetscInt n_species;
            PetscInt n_states_global;
            PetscInt n_local_states;
            arma::Mat<PetscInt> local_states;
            arma::Mat<PetscInt> stoichiometry;

            PetscLayout vec_layout = nullptr;
            AO lex2petsc = nullptr;

            Zoltan_Struct *zoltan;
        public:

            // Generic Interface
            explicit FiniteStateSubset(MPI_Comm new_comm);

            MPI_Comm GetComm();

            arma::Row<PetscInt> GetFSPSize();

            void SetStoichiometry(arma::Mat<PetscInt> SM);

            void SetSize(arma::Row<PetscInt> &new_fsp_size);

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

            friend arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

            /* Zoltan interface functions */
            friend int zoltan_num_obj(void *fss_data, int *ierr);

            friend void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                                 ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                 int *ierr);

            friend int zoltan_num_edges (void *data, int num_gid_entries, int num_lid_entries,
                    ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);
            friend void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                    ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim, float *ewgts, int *ierr);
            friend void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
            friend void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                              int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);
        };

        arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

        /* Zoltan interface functions */
        int zoltan_num_obj(void *fss_data, int *ierr);
        void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                             int *ierr);
        int zoltan_num_edges (void *data, int num_gid_entries, int num_lid_entries,
                                     ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);
        void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                     ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim, float *ewgts, int *ierr);
        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                   int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);
        /*!
         * Helper functions to convert back and forth between partitioning options and string
         */
         std::string part2str(PartitioningType part);
         PartitioningType str2part(std::string str);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSET_H
