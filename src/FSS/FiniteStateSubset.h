//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSET_H
#define PARALLEL_FSP_FINITESTATESUBSET_H

#include <zoltan.h>
#include "util/cme_util.h"
#include "FiniteStateSubsetZoltanQuery.h"

namespace cme {
    namespace parallel {
        enum PartitioningType {
            Naive, RCB, Graph, HyperGraph, Hierarch, NotSet
        };

        enum PartitioningApproach {
            FromScratch, Repartition, Refine
        };

        /// Base class for the Finite State Subset object.
        /**
         * The Finite State Subset object contains the data and methods related to the storage, management, and parallel
         * distribution of the states included by the Finite State Projection algorithm. Our current implementation relies on
         * Zoltan's dynamic load-balancing tools for the parallel distribution of these states into the processors.
         * We use the same graph and hypergraph models of parallel spMxV proposed in [Catalyurek et al., IEEE Trans Parallel Dist Sys, Vol 10, No 7, 1999]
         * but adjust object's weights to reflect better the data structure we use for the time-dependent CME matrix.
         * */
        class FiniteStateSubset {
        protected:
            int set_up = 0;
            int stoich_set = 0;

            MPI_Comm comm;
            PartitioningType partitioning_type;
            PartitioningApproach repart_approach = FromScratch;

            arma::Row<PetscInt> fsp_size;
            arma::Mat<PetscInt> stoichiometry;

            PetscInt n_species;
            PetscInt n_states_global;
            PetscInt n_local_states;
            arma::Mat<PetscInt> local_states;

            /// For event logging
            PetscLogEvent generate_graph_data;
            PetscLogEvent call_partitioner;

            void LocalStatesFromAO();

            PetscLayout vec_layout = nullptr;
            AO lex2petsc = nullptr;

            arma::Mat<PetscInt>
            compute_my_naive_local_states(); ///< Compute the local states owned by the processor in the naive partitioning. This is used as the initial guess for the partitioning algorithms.

            /// Struct to stores data for graph/hypergraph partitioning algorithms
            /**
             * This data structure corresponds to the row-wise decomposition of the CME matrix. Each object/vertex corresponds
             * to a state of the CME. Connectivity between states are represented by edges in the graph model (using the generalized graph model
             * of Catalyurek et al.), and hyper-edges in
             * the hypergraph model (using the column-net model of Catalyurek et al.).
             * The details about these graph and hypergraph models could be found in Catalyurek et al., IEEE Trans Parallel Dist Sys, Vol 10, No 7, 1999.
             * We adjust the computational weights to reflect our specific implementation of the time-dependent CME matrix [Vo & Munsky, 2019?].
             */
            struct ConnectivityData {
                AO lex2zoltan; ///< Store ordering from FSP states' natural indexing to Zoltan's indexing

                int num_local_states; ///< Number of local states held by the processor in the current partitioning
                PetscInt *states_gid; ///< Global IDs of the states held by the processor, these IDs are the lexicographic orders of the states in the hyper-rectangular FSP
                float *states_weights; ///< Computational weights associated with each state, here we assign to these weights the number of FLOPs needed

                int *num_edges; ///< Number of states that share information with each local states

                int num_reachable_states; ///< Number of nz entries on the rows of the FSP matrix corresponding to local states
                PetscInt *reachable_states; ///< Global indices of nz entries on the rows corresponding to local states
                int *reachable_states_proc; ///< Processors that own the reachable states
                float *edge_weights; ///< For storing the edge weights in graph model
                int *edge_ptr; ///< reachable_states[edge_ptr[i] to ege_ptr[i+1]-1] contains the ids of states connected to local state i
            } adj_data;

            void GenerateVertexData(arma::Mat<PetscInt> &local_states_tmp);

            void FreeVertexData();

            void GenerateGraphData(arma::Mat<PetscInt> &local_states_tmp); ///< The name says it all

            void FreeGraphData(); ///< The name says it all

            void GenerateHyperGraphData(arma::Mat<PetscInt> &local_states_tmp); ///< The name says it all

            void FreeHyperGraphData(); ///< The name says it all

            struct GeomData {
                int dim;
                double *states_coo;
            } geom_data;

            void GenerateGeomData(arma::Row<PetscInt> &fsp_size, arma::Mat<PetscInt> &local_states_tmp);

            void FreeGeomData();

            // These variables are needed for partitioning with Zoltan
            Zoltan_Struct *zoltan;
            std::string zoltan_part_opt = std::string("PARTITION");
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

            friend int zoltan_obj_size(
                    void *data,
                    int num_gid_entries,
                    int num_lid_entries,
                    ZOLTAN_ID_PTR global_id,
                    ZOLTAN_ID_PTR local_id,
                    int *ierr);

            friend arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

        public:

            // Generic Interface
            explicit FiniteStateSubset(MPI_Comm new_comm);

            void SetStoichiometry(arma::Mat<PetscInt> SM);

            void SetSize(arma::Row<PetscInt> new_fsp_size);

            void SetRepartApproach(PartitioningApproach approach);

            MPI_Comm GetComm();

            arma::Row<PetscInt> GetFSPSize();

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
        };

        /// Compute the marginal distributions from a given finite state subset and parallel probability vector
        arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

        /*
         * Helper functions to convert back and forth between partitioning options and string
         */
        std::string part2str(PartitioningType part);

        PartitioningType str2part(std::string str);

        std::string partapproach2str(PartitioningApproach part_approach);

        PartitioningApproach str2partapproach(std::string str);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSET_H
