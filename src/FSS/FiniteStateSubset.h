//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSET_H
#define PARALLEL_FSP_FINITESTATESUBSET_H

#include <zoltan.h>
#include <petscis.h>
#include "util/cme_util.h"
#include "FiniteStateSubsetZoltanQuery.h"

namespace cme {
    namespace parallel {
        enum PartitioningType {
            Graph, HyperGraph
        };

        enum PartitioningApproach {
            FromScratch, Repartition
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
            static const int hash_table_length = 1000000;

            int set_up = 0;
            int stoich_set = 0;

            MPI_Comm comm;
            PartitioningType partitioning_type = Graph;
            PartitioningApproach repart_approach = Repartition;

            arma::Mat<PetscInt> stoichiometry;

            PetscInt n_species;
            PetscInt n_reactions;
            PetscInt nstate_global;
            PetscInt nstate_local;

            /// Left and right hand sides of the FSP constraints
            std::function<void(PetscInt i, PetscInt num_states, PetscInt num_species, PetscInt *states,
                               double *vals)> lhs_constr;
            arma::Row<double> rhs_constr;

            /// For event logging
            PetscLogEvent generate_graph_data;
            PetscLogEvent call_partitioner;

            /// Armadillo array to store states owned by this processor
            /**
             * States are stored as column vectors of integers.
             */
            arma::Mat<PetscInt> local_states;

            /// Local states status
            /**
             * Its length must be the same as the number of columns in local_states.
             * State's status is either 'active'
             * (1, to be expanded in state space exploration) or 'inactive' (0).
             */
            arma::Row<PetscInt> local_states_status;

            /// Armadillo array to store states reachable from the local states, alongside their statuses
            /**
             * The first N*M rows store reachable states, with N = number of species and M = number of reactions.
             * Entries (i*N .. (i+1)*N-1, j) storing the state reachable
             * from the j-th local state via the i-th chemical reaction.
             */
            arma::Mat<PetscInt> local_reachable_states;

            /// Armadillo array to store the status of the reachable states
            /**
             * Status = i if the (i-1)-th constraint was not satisfied. Otherwise status = 0.
             */
            arma::Mat<PetscInt> local_reachable_states_status;

            /// PETSc vector layout for the probability distribution
            PetscLayout state_layout = nullptr;

            /// Variable to store local ids of frontier states
            arma::uvec frontier_lids;

            /// Zoltan directory
            /**
             * This is essentially a parallel hash table. We use it to store existing states for fast lookup.
             */
            Zoltan_DD_Struct *state_directory;

            /// Zoltan struct for load-balancing
            Zoltan_Struct *zoltan_lb;

            /// Zoltan struct for load-balancing the state space search
            Zoltan_Struct *zoltan_explore;

            std::string zoltan_part_opt = std::string("PARTITION");

            /* Private functions */

            /// Initialize the number of GIDs and the Load-balancing method in Zoltan
            /**
             * Call level: collective.
             */
            void InitZoltanParameters();

            /// Distribute the frontier states to all processors for state space exploration
            /**
             * Call level: collective.
             */
            void DistributeFrontier();

            /// Distribute the frontier states to all processors in FSS's communication context for state space exploration
            /**
             * Call level: collective.
             */
            void LoadBalance();

            /// Add a set of states to the global and local state set
            /**
             * Call level: collective.
             * @param X : armadillo matrix of states to be added. X.n_rows = number of species. Each processor input its own
             * local X. Different input sets from different processors may overlap.
             */
            void AddStates(arma::Mat<PetscInt>& X);

            /// Check if a state satisfies all constraints
            /**
             *
             * @param x
             * @return 0 if x satisfies all constraints; otherwise i+1 where i is the index of the first constraint violated by x.
             */
            inline int CheckConstraints(arma::Col<PetscInt>& x);

            /* Zoltan interface functions */
            friend int zoltan_num_obj(void *fss_data, int *ierr);

            friend void zoltan_obj_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                                        ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                        int *ierr);

            friend int zoltan_obj_size(
                    void *data,
                    int num_gid_entries,
                    int num_lid_entries,
                    ZOLTAN_ID_PTR global_id,
                    ZOLTAN_ID_PTR local_id,
                    int *ierr);

            friend void
            zoltan_pack_states(void *data, int num_gid_entries, int num_lid_entries, int num_ids,
                               ZOLTAN_ID_PTR global_ids,
                               ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf, int *ierr);

            friend void
            zoltan_unpack_states(void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes,
                                 int *idx, char *buf, int *ierr);

            friend int zoltan_num_frontier(void *fss_data, int *ierr);

            friend void zoltan_frontier_list(void *fss_data, int num_gid_entries, int num_lid_entries,
                                      ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                      int *ierr);

//            friend int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries,
//                                        ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr);
//            friend void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
//                                         ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs,
//                                         int wgt_dim, float *ewgts, int *ierr);
//            friend void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
//            friend void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
//                                              int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr,
//                                              ZOLTAN_ID_PTR pin_gid, int *ierr);


            friend arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

        public:

            // Generic Interface
            explicit FiniteStateSubset(MPI_Comm new_comm, PetscInt num_species);

            void SetStoichiometry(arma::Mat<PetscInt> SM);

            void SetShape(std::function<void(PetscInt, PetscInt, PetscInt, PetscInt *, double *)> &lhs_fun,
                          arma::Row<double> &rhs_bounds);

            void SetShapeBounds(arma::Row<PetscInt> &rhs_bounds);

            void SetShapeBounds(arma::Row<double> &rhs_bounds);

            void SetLBType(PartitioningType lb_type);

            void SetRepartApproach(PartitioningApproach approach);

            /// Set the initial states.
            /**
             * Call level: collective.
             * Each processor enters its own set of initial states. Initial state could be empty, but at least one processor
             * must insert at least one state. Initial states from different processors must not overlap.
             */
            void SetInitialStates(arma::Mat<PetscInt> X0);

            void GenerateStatesAndOrdering();

            arma::Row<PetscInt> State2Petsc(arma::Mat<PetscInt> state);

            void State2Petsc(arma::Mat<PetscInt> state, PetscInt *indx);

            arma::Row<PetscReal> SinkStatesReduce(Vec P);

            MPI_Comm GetComm();

            arma::Row<double> GetShapeBounds();

            PetscInt GetNumLocalStates();

            PetscInt GetNumGlobalStates();

            PetscInt GetNumSpecies();

            PetscInt GetNumReactions();

            arma::Mat<PetscInt> GetLocalStates();

            std::tuple<PetscInt, PetscInt> GetLayoutStartEnd();

            void Destroy();

            ~FiniteStateSubset();
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
