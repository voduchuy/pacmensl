//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSET_H
#define PARALLEL_FSP_FINITESTATESUBSET_H

#include <zoltan.h>
#include <petscis.h>

#include "Partitioner/StatePartitioner.h"
#include "util/cme_util.h"

namespace cme {
    namespace parallel {
        typedef void fsp_constr_multi_fn( int n_species, int n_constraints, int n_states, int *states, int *output );

        struct FiniteStateSubsetLogger {
            /// For event logging
            PetscLogEvent state_exploration_event;
            PetscLogEvent check_constraints_event;
            PetscLogEvent add_states_event;
            PetscLogEvent call_partitioner_event;
            PetscLogEvent zoltan_dd_stuff_event;
            PetscLogEvent total_update_dd_event;
            PetscLogEvent distribute_frontiers_event;

            void register_all( MPI_Comm comm );

            void event_begin( PetscLogEvent event );

            void event_end( PetscLogEvent event );
        };

        /// Base class for the Finite State Subset object.
        /**
         * The Finite State Subset object contains the data and methods related to the storage, management, and parallel
         * distribution of the states included by the Finite State Projection algorithm. Our current implementation relies on
         * Zoltan's dynamic load-balancing tools for the parallel distribution of these states into the processors.
         * */
        class StateSetBase {
        protected:
            static const int hash_table_length_ = 1000000;

            MPI_Comm comm_;

            int set_up_ = 0;
            int stoich_set_ = 0;

            int comm_size_;
            int my_rank_;

            arma::Mat< int > stoichiometry_matrix_;
            double lb_threshold_ = 0.2;

            int num_species_;
            int num_reactions_;
            int num_global_states_;
            int num_local_states_ = 0;
            int num_global_states_old_ = 0;

            FiniteStateSubsetLogger logger_;

            StatePartitioner partitioner_;

            /// Armadillo array to store states owned by this processor
            /**
             * States are stored as column vectors of integers.
             */
            arma::Mat< int > local_states_;
            arma::Row< char > local_states_status_;

            arma::Row< int > state_layout_;
            arma::Row< int > ind_starts_;

            /// Variable to store local ids of frontier states
            arma::uvec frontier_lids_;
            arma::Mat< int > local_frontier_gids_;
            arma::Mat< int > frontiers_;


            /// Zoltan directory
            /**
             * This is essentially a parallel hash table. We use it to store existing states for fast lookup.
             */
            Zoltan_DD_Struct *state_directory_;

            /// Zoltan struct for load-balancing the state space search
            Zoltan_Struct *zoltan_explore_;

            void init_zoltan_parameters( );

            void distribute_frontiers( );

            void load_balance( );

            void update_layout( );

            void update_state_indices( );

            void update_state_status( arma::Mat< PetscInt > states, arma::Row< char > status );

            void update_state_indices_status( arma::Mat< PetscInt > states, arma::Row< PetscInt > local_ids,
                                              arma::Row< char > status );

            void retrieve_state_status( );

            // TODO: frontier distribution use multi-dimensional gids
            static int zoltan_num_frontier( void *data, int *ierr );

            static void zoltan_frontier_list( void *data, int num_gid_entries, int num_lid_entries,
                                              ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim,
                                              float *obj_wgts,
                                              int *ierr );

            static int
            zoltan_frontier_size( void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                  ZOLTAN_ID_PTR local_id, int *ierr );

            static void
            pack_frontiers( void *data, int num_gid_entries, int num_lid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                            ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf, int *ierr );

            static void
            unpack_frontiers( void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes,
                              int *idx,
                              char *buf, int *ierr );

            static void mid_frontier_migration( void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                                ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                                int *import_procs,
                                                int *import_to_part, int num_export, ZOLTAN_ID_PTR export_global_ids,
                                                ZOLTAN_ID_PTR export_local_ids, int *export_procs, int *export_to_part,
                                                int *ierr );

        public:
            // Not copyable or movable
            StateSetBase( const StateSetBase & ) = delete;

            StateSetBase &operator=( const StateSetBase & )
            = delete;

            // Generic Interface
            explicit StateSetBase( MPI_Comm new_comm, int num_species, PartitioningType lb_type = Graph,
                                   PartitioningApproach lb_approach = Repartition );

            void set_stoichiometry( arma::Mat< int > SM );

            void set_initial_states( arma::Mat< PetscInt > X0 );

            void add_states( const arma::Mat< int > &X );

            arma::Row< PetscInt > state2ordering( arma::Mat< PetscInt > &state ) const;

            void state2ordering( arma::Mat< PetscInt > &state, PetscInt *indx ) const;

            virtual void expand( ) = 0;

            /// Getters
            MPI_Comm get_comm( ) const;

            int get_num_local_states( ) const;

            int get_num_global_states( ) const;

            int get_num_species( ) const;

            int get_num_reactions( ) const;

            const arma::Mat< int >& get_states_ref () const;

            arma::Mat< int > copy_states_on_proc( ) const;

            std::tuple< int, int > get_ordering_ends_on_proc( ) const;

            ~StateSetBase( );
        };


        /// Compute the marginal distributions from a given finite state subset and parallel probability vector
        arma::Col< PetscReal > marginal( StateSetBase &set, Vec P, PetscInt species );
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSET_H