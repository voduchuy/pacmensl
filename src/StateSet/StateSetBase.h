//
// Created by Huy Vo on 12/4/18.
//

#ifndef PACMENSL_FINITESTATESUBSET_H
#define PACMENSL_FINITESTATESUBSET_H

#include <zoltan.h>
#include <petscis.h>

#include "StatePartitioner.h"
#include "Sys.h"

namespace pacmensl {

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
    public: NOT_COPYABLE_NOT_MOVABLE( StateSetBase );

        explicit StateSetBase( MPI_Comm new_comm, int num_species, PartitioningType lb_type = GRAPH,
                               PartitioningApproach lb_approach = REPARTITION );

        /// Set the stoichiometry matrix using Armadillo's matrix class.
        /**
         * Call level: collective.
         * The user must ensure that all processors must enter the same stoichiometry matrix.
         */
        void SetStoichiometryMatrix( const arma::Mat< int > &SM );

        /// Set the stoichiometry matrix using C array.
        /**
         * Call level: collective.
         * @param num_species : number of species
         * @param num_reactions : number of reactions
         * @param values : array of stoichiometry entries, where values[num_species*i],..,values[num_species*(i+1)-1] store the entries corresponding to the i-th reaction.
         * The user must ensure that all processors must enter the same stoichiometry matrix.
         */
        void SetStoichiometryMatrixC( int num_species, int num_reactions, const int *values );

        /// Set the initial states.
        /**
         * Call level: collective.
         * Each processor enters its own set of initial states. Initial state could be empty, but at least one processor
         * must insert at least one state. Initial states from different processors must not overlap.
         */
        void SetInitialStates( arma::Mat< PetscInt > X0 );

        /// Set the initial states.
        /**
         * Call level: collective.
         * @param num_states: number of states.
         * @param vals: array of states, where vals[i*num_states, .. (i-1)*num_states-1] stores entries of the i-th state.
         * Each processor enters its own set of initial states. Initial state could be empty, but at least one processor
         * must insert at least one state. The user must ensure that vals have num_species*num_states entries, where num_species
         * is the number of species the calling StateSetBase class was constructed with.
         */
        void SetInitialStates( int num_states, int* vals );

        /// Add a set of states to the global and local state set
        /**
         * Call level: collective.
         * @param X : armadillo matrix of states to be added. X.n_rows = number of species. Each processor input its own
         * local X. Different input sets from different processors may overlap.
         */
        void AddStates( const arma::Mat< int > &X );

        /// Add a set of states to the global and local state set
        /**
         * Call level: collective.
         * @param num_states: number of states.
         * @param vals: array of states, where vals[i*num_states, .. (i-1)*num_states-1] stores entries of the i-th state.
         * Each processor enters its own set of initial states. The user must ensure that vals have num_species*num_states entries, where num_species
         * is the number of species the calling StateSetBase class was constructed with.
         */
        void AddStates( int num_states, int* vals);

        arma::Row< PetscInt > State2Index( arma::Mat< PetscInt > &state ) const;

        void State2Index( arma::Mat< PetscInt > &state, int *indx ) const;

        void State2Index( int num_states, const int* state, int *indx) const;

        virtual void Expand( ) {};

        MPI_Comm GetComm( ) const;

        int GetNumLocalStates( ) const;

        int GetNumGlobalStates( ) const;

        int GetNumSpecies( ) const;

        int GetNumReactions( ) const;

        /// Get access to the list of states stored in the calling processor.
        /**
         * Call level: not collective.
         * Note: The reference is for read-only purpose.
         * @return const reference to the armadillo matrix that stores the states on the calling processor. Each column represents a state.
         */
        const arma::Mat< int > &GetStatesRef( ) const;

        /// Copy the list of states stored in the calling processor.
        /**
         * Call level: not collective.
         * @return armadillo matrix that stores the states on the calling processor. Each column represents a state.
         */
        arma::Mat< int > CopyStatesOnProc( ) const;

        /// Copy the list of states stored in the calling processor into a C array.
        /**
         * Call level: not collective.
         * @param num_local_states : (in/out) number of local states.
         * @param state_array : pointer to the copied states.
         * NOTE: do not allocate memory for state_array since this will be done within the function.
         */
        void CopyStatesOnProc( int num_local_states, int* state_array) const;

        std::tuple< int, int > GetOrderingStartEnd( ) const;

        ~StateSetBase( );

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

        static int zoltan_num_frontier( void *data, int *ierr );

        static void zoltan_frontier_list( void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                          ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr );

        static int zoltan_frontier_size( void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                         ZOLTAN_ID_PTR local_id, int *ierr );

        static void
        pack_frontiers( void *data, int num_gid_entries, int num_lid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                        ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf, int *ierr );

        static void
        unpack_frontiers( void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes, int *idx,
                          char *buf, int *ierr );

        static void mid_frontier_migration( void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                            ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                            int *import_procs, int *import_to_part, int num_export,
                                            ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
                                            int *export_procs, int *export_to_part, int *ierr );
    };
}

#endif //PACMENSL_FINITESTATESUBSET_H
