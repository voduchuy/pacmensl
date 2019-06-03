//
// Created by Huy Vo on 12/4/18.
//
#include "StateSetBase.h"


namespace cme {
    namespace parallel {
        StateSetBase::StateSetBase( MPI_Comm new_comm, int _num_species, PartitioningType lb_type,
                                    PartitioningApproach lb_approach ) : partitioner_( new_comm ) {
            int ierr;
            MPI_Comm_dup( new_comm, &comm_ );
            MPI_Comm_size( comm_, &comm_size_ );
            MPI_Comm_rank( comm_, &my_rank_ );

            state_layout_.set_size( comm_size_ );
            ind_starts_.set_size( comm_size_ );
            state_layout_.fill( 0 );

            num_species_ = _num_species;
            local_states_.resize( num_species_, 0 );
            num_global_states_ = 0;
            stoichiometry_matrix_.resize( num_species_, 0 );

            zoltan_explore_ = Zoltan_Create( comm_ );
            ierr = Zoltan_DD_Create( &state_directory_, comm_, num_species_, 1, 1, hash_table_length_, 0 );
            ZOLTANCHKERRABORT( comm_, ierr );

            partitioner_.set_up( lb_type, lb_approach );

            init_zoltan_parameters();

            logger_.register_all( comm_ );
        };

        void StateSetBase::set_stoichiometry( arma::Mat< int > SM ) {
            stoichiometry_matrix_ = SM;
            num_species_ = ( int ) SM.n_rows;
            num_reactions_ = ( int ) SM.n_cols;
            stoich_set_ = 1;
        }


        /// Set the initial states.
        /**
         * Call level: collective.
         * Each processor enters its own set of initial states. Initial state could be empty, but at least one processor
         * must insert at least one state. Initial states from different processors must not overlap.
         */
        void StateSetBase::set_initial_states( arma::Mat< int > X0 ) {
            int my_rank;
            MPI_Comm_rank( comm_, &my_rank );

            if ( X0.n_rows != num_species_ ) {
                throw std::runtime_error(
                        "set_initial_states: number of rows in input array is not the same as the number of species.\n" );
            }

            PetscPrintf( comm_, "Adding initial states...\n" );
            add_states( X0 );
            PetscPrintf( comm_, "Initial states set...\n" );
        }

        const arma::Mat< int > &StateSetBase::get_states_ref( ) const {
            return local_states_;
        }

        arma::Mat< int > StateSetBase::copy_states_on_proc( ) const {
            arma::Mat< int > states_return( num_species_, num_local_states_ );
            for ( auto j{0}; j < num_local_states_; ++j ) {
                for ( auto i{0}; i < num_species_; ++i ) {
                    states_return( i, j ) = ( int ) local_states_( i, j );
                }
            }
            return states_return;
        }

        StateSetBase::~StateSetBase( ) {
            int ierr;
            ierr = MPI_Comm_free( &comm_ );
            MPICHKERRABORT( comm_, ierr );
            Zoltan_Destroy( &zoltan_explore_ );
            Zoltan_DD_Destroy( &state_directory_ );
        }

        /// Distribute the frontier states to all processors for state space exploration

        /**
         * Call level: collective.
         */
        void StateSetBase::distribute_frontiers( ) {
            PetscLogEventBegin( logger_.distribute_frontiers_event, 0, 0, 0, 0 );

            local_frontier_gids_ = state2ordering( frontiers_ );

            // Variables to store Zoltan's output
            int zoltan_err, ierr;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            zoltan_err = Zoltan_LB_Partition( zoltan_explore_, &changes, &num_gid_entries, &num_lid_entries,
                                              &num_import, &import_global_ids, &import_local_ids, &import_procs,
                                              &import_to_part, &num_export, &export_global_ids, &export_local_ids,
                                              &export_procs, &export_to_part );
            ZOLTANCHKERRABORT( comm_, zoltan_err );

            Zoltan_LB_Free_Part( &import_global_ids, &import_local_ids, &import_procs, &import_to_part );
            Zoltan_LB_Free_Part( &export_global_ids, &export_local_ids, &export_procs, &export_to_part );
            PetscLogEventEnd( logger_.distribute_frontiers_event, 0, 0, 0, 0 );
        }
        /// Distribute the frontier states to all processors in FSS's communication context for state space exploration

        /**
         * Call level: collective.
         */
        void StateSetBase::load_balance( ) {
            partitioner_.partition( local_states_, state_directory_, stoichiometry_matrix_, &state_layout_[ 0 ] );
            num_local_states_ = local_states_.n_cols;
            PetscMPIInt nslocal = num_local_states_;
            PetscMPIInt nsglobal;
            MPI_Allreduce( &nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm_ );
            num_global_states_ = nsglobal;

            // Update state global ids
            update_layout( );
            update_state_indices( );
        }
        /// Add a set of states to the global and local state set

        /**
         * Call level: collective.
         * @param X : armadillo matrix of states to be added. X.n_rows = number of species. Each processor input its own
         * local X. Different input sets from different processors may overlap.
         */
        void StateSetBase::add_states( const arma::Mat< int > &X ) {
            PetscLogEventBegin( logger_.add_states_event, 0, 0, 0, 0 );
            int zoltan_err;

            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids; //
            arma::Row< int > owner( X.n_cols );
            arma::Row< int > parts( X.n_cols );
            arma::uvec iselect;


            // Probe if states in X are already owned by some processor
            local_dd_gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE>>::from( X );
            zoltan_err = Zoltan_DD_Find( state_directory_, &local_dd_gids[ 0 ], nullptr, nullptr, nullptr, X.n_cols,
                                         &owner[ 0 ] );
            iselect = arma::find( owner == -1 );
            ZOLTANCHKERRABORT( comm_, zoltan_err );

            // Shed any states that are already included
            arma::Mat< int > Xadd = X.cols( iselect );
            local_dd_gids = local_dd_gids.cols( iselect );

            // Add local states to Zoltan directory (only 1 state from 1 processor will be added in case of overlapping)
            parts.resize( Xadd.n_cols );
            parts.fill( my_rank_ );
            PetscLogEventBegin( logger_.zoltan_dd_stuff_event, 0, 0, 0, 0 );
            zoltan_err = Zoltan_DD_Update( state_directory_, &local_dd_gids[ 0 ], nullptr, nullptr, parts.memptr( ),
                                           Xadd.n_cols );
            ZOLTANCHKERRABORT( comm_, zoltan_err );
            PetscLogEventEnd( logger_.zoltan_dd_stuff_event, 0, 0, 0, 0 );

            // Remove overlaps between processors
            parts.resize( Xadd.n_cols );
            zoltan_err = Zoltan_DD_Find( state_directory_, &local_dd_gids[ 0 ], nullptr, nullptr, parts.memptr( ),
                                         Xadd.n_cols, nullptr );
            ZOLTANCHKERRABORT( comm_, zoltan_err );

            iselect = arma::find( parts == my_rank_ );
            Xadd = Xadd.cols( iselect );

            arma::Row< char > X_status( Xadd.n_cols );
            if ( Xadd.n_cols > 0 ) {
                // Append X to local states
                X_status.fill( 1 );
                local_states_ = arma::join_horiz( local_states_, Xadd );
            }

            int nstate_local_old = num_local_states_;
            num_local_states_ = ( int ) local_states_.n_cols;
            PetscMPIInt nslocal = num_local_states_;
            PetscMPIInt nsglobal;
            MPI_Allreduce( &nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm_ );
            num_global_states_ = nsglobal;

            // Update layout_
            update_layout( );

            // Update local ids and status
            arma::Row< int > local_ids;
            if ( num_local_states_ > nstate_local_old ) {
                local_ids = arma::regspace< arma::Row< int>>( nstate_local_old, num_local_states_ - 1 );
            } else {
                local_ids.resize( 0 );
            }

            update_state_indices_status( Xadd, local_ids, X_status );
            PetscLogEventEnd( logger_.add_states_event, 0, 0, 0, 0 );
        }
        /// Check if a state satisfies all constraints

        /// Check if a list of states satisfy a constraint

        /// Initialize the number of GIDs and the Load-balancing method in Zoltan

        /**
         * Call level: collective.
         */
        void StateSetBase::init_zoltan_parameters( ) {
            // Parameters for state exploration load-balancing
            Zoltan_Set_Param( zoltan_explore_, "NUM_GID_ENTRIES", "1" );
            Zoltan_Set_Param( zoltan_explore_, "NUM_LID_ENTRIES", "1" );
            Zoltan_Set_Param( zoltan_explore_, "IMBALANCE_TOL", "1.01" );
            Zoltan_Set_Param( zoltan_explore_, "AUTO_MIGRATE", "1" );
            Zoltan_Set_Param( zoltan_explore_, "RETURN_LISTS", "ALL" );
            Zoltan_Set_Param( zoltan_explore_, "DEBUG_LEVEL", "0" );
            Zoltan_Set_Param( zoltan_explore_, "LB_METHOD", "Block" );
            Zoltan_Set_Num_Obj_Fn( zoltan_explore_, &StateSetBase::zoltan_num_frontier, ( void * ) this );
            Zoltan_Set_Obj_List_Fn( zoltan_explore_, &StateSetBase::zoltan_frontier_list, ( void * ) this );
            Zoltan_Set_Obj_Size_Fn( zoltan_explore_, &StateSetBase::zoltan_frontier_size, ( void * ) this );
            Zoltan_Set_Pack_Obj_Multi_Fn( zoltan_explore_, &StateSetBase::pack_frontiers, ( void * ) this );
            Zoltan_Set_Mid_Migrate_PP_Fn( zoltan_explore_, &StateSetBase::mid_frontier_migration, ( void * ) this );
            Zoltan_Set_Unpack_Obj_Multi_Fn( zoltan_explore_, &StateSetBase::unpack_frontiers, ( void * ) this );
        }

        void StateSetBase::update_layout( ) {
            state_layout_[ my_rank_ ] = num_local_states_;
            MPI_Allgather( &state_layout_[ my_rank_ ], 1, MPI_INT, &state_layout_[ 0 ], 1, MPI_INT, comm_ );
            ind_starts_[ 0 ] = 0;
            for ( int i{1}; i < comm_size_; ++i ) {
                ind_starts_[ i ] = ind_starts_[ i - 1 ] + state_layout_[ i - 1 ];
            }
        }

        /// Generate the indices of the states in the Petsc vector ordering.

        /**
         * Call level: collective.
         * @param state: matrix of input states. Each column represent a state. Each processor inputs its own set of states.
         * @return Armadillo row vector of indices. The index of each state is nonzero of the state is a member of the finite state subset. Otherwise, the index is -1 (if state does not exist in the subset, or some entries of the state are 0) or -2-i if the state violates constraint i.
         */
        arma::Row< int > StateSetBase::state2ordering( arma::Mat< int > &state ) const {
            arma::Row< int > indices( state.n_cols );
            indices.fill( 0 );

            for ( int i{0}; i < state.n_cols; ++i ) {
                for ( int j = 0; j < num_species_; ++j ) {
                    if ( state( j, i ) < 0 ) {
                        indices( i ) = -1;
                        break;
                    }
                }
            }

            arma::uvec i_vaild_constr = arma::find( indices == 0 );

            arma::Row< ZOLTAN_ID_TYPE > state_indices( i_vaild_constr.n_elem );
            arma::Row< int > parts( i_vaild_constr.n_elem );
            arma::Row< int > owners( i_vaild_constr.n_elem );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE >>::from(
                    state.cols( i_vaild_constr ));

            Zoltan_DD_Find( state_directory_, gids.memptr( ), state_indices.memptr( ), nullptr, parts.memptr( ),
                            i_vaild_constr.n_elem, owners.memptr( ));

            for ( int i{0}; i < i_vaild_constr.n_elem; i++ ) {
                int indx = i_vaild_constr( i );
                if ( owners[ i ] >= 0 && parts[ i ] >= 0 ) {
                    indices( indx ) = ind_starts_( parts[ i ] ) + state_indices( i );
                } else {
                    indices( indx ) = -1;
                }
            }

            return indices;
        }
        /// Generate the indices of the states in the Petsc vector ordering.

        /**
         * Call level: collective.
         * @param state: matrix of input states. Each column represent a state. Each processor inputs its own set of states.
         * indx: output array of indices.
         * @return none.
         */
        void StateSetBase::state2ordering( arma::Mat< int > &state, int *indx ) const {

            arma::Row< int > ipositive( state.n_cols );
            ipositive.fill( 0 );

            for ( int i{0}; i < ipositive.n_cols; ++i ) {
                for ( int j = 0; j < num_species_; ++j ) {
                    if ( state( j, i ) < 0 ) {
                        ipositive( i ) = -1;
                        indx[ i ] = -1;
                        break;
                    }
                }
            }

            arma::uvec i_vaild_constr = arma::find( ipositive == 0 );

            arma::Row< ZOLTAN_ID_TYPE > state_indices( i_vaild_constr.n_elem );
            arma::Row< int > parts( i_vaild_constr.n_elem );
            arma::Row< int > owners( i_vaild_constr.n_elem );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE >>::from(
                    state.cols( i_vaild_constr ));

            Zoltan_DD_Find( state_directory_, gids.memptr( ), state_indices.memptr( ), nullptr, parts.memptr( ),
                            i_vaild_constr.n_elem, owners.memptr( ));

            for ( int i{0}; i < i_vaild_constr.n_elem; i++ ) {
                auto ii = i_vaild_constr( i );
                if ( owners[ i ] >= 0 && parts[ i ] >= 0 ) {
                    indx[ ii ] = ind_starts_( parts[ i ] ) + state_indices( i );
                } else {
                    indx[ ii ] = -1;
                }
            }
        }

        std::tuple< int, int > StateSetBase::get_ordering_ends_on_proc( ) const {
            int start, end, ierr;
            start = 0;
            for ( int i{0}; i < my_rank_; ++i ) {
                start += state_layout_[ i ];
            }
            end = start + state_layout_[ my_rank_ ];
            return std::make_tuple( start, end );
        }


        MPI_Comm StateSetBase::get_comm( ) const {
            return comm_;
        }


        int StateSetBase::get_num_local_states( ) const {
            return num_local_states_;
        }

        int StateSetBase::get_num_global_states( ) const {
            return num_global_states_;
        }

        int StateSetBase::get_num_species( ) const {
            return ( int( num_species_ ));
        }

        int StateSetBase::get_num_reactions( ) const {
            return stoichiometry_matrix_.n_cols;
        }

        /// Update the distributed directory of states (after FSP expansion, Load-balancing...)
        /**
         * Call lvel: collective.
         */
        void StateSetBase::update_state_indices( ) {
            logger_.event_begin( logger_.total_update_dd_event );
            int zoltan_err;
            // Update hash table
            auto local_dd_gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE>>::from( local_states_ );
            arma::Row< ZOLTAN_ID_TYPE > lids;
            if ( num_local_states_ > 0 ) {
                lids = arma::regspace< arma::Row< ZOLTAN_ID_TYPE>>( 0, num_local_states_ - 1 );
            } else {
                lids.set_size( 0 );
            }
            arma::Row< int > parts( num_local_states_ );
            parts.fill( my_rank_ );
            zoltan_err = Zoltan_DD_Update( state_directory_, &local_dd_gids[ 0 ], &lids[ 0 ], nullptr, parts.memptr( ),
                                           num_local_states_ );
            ZOLTANCHKERRABORT( comm_, zoltan_err );
            logger_.event_end( logger_.total_update_dd_event );
        }

        void StateSetBase::update_state_status( arma::Mat< int > states, arma::Row< char > status ) {
            logger_.event_begin( logger_.total_update_dd_event );
            int zoltan_err;

            // Update hash table
            int n_update = status.n_elem;
            if ( n_update != states.n_cols ) {
                PetscPrintf( comm_,
                             "FSS: update_state_status: states and status arrays have incompatible dimensions.\n" );
            }
            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids( num_species_, n_update );

            if ( n_update > 0 ) {
                for ( int i = 0; i < n_update; ++i ) {
                    local_dd_gids.col( i ) = arma::conv_to< arma::Col< ZOLTAN_ID_TYPE>>::from( states.col( i ));
                }
            }

            zoltan_err = Zoltan_DD_Update( state_directory_, local_dd_gids.memptr( ), nullptr, status.memptr( ),
                                           nullptr, n_update );
            ZOLTANCHKERRABORT( comm_, zoltan_err );
            logger_.event_end( logger_.total_update_dd_event );
        }


        void StateSetBase::update_state_indices_status( arma::Mat< int > states, arma::Row< int > local_ids,
                                                        arma::Row< char > status ) {
            logger_.event_begin( logger_.total_update_dd_event );
            int zoltan_err;

            // Update hash table
            int n_update = local_ids.n_elem;
            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids( num_species_, n_update );
            arma::Row< ZOLTAN_ID_TYPE > lids( n_update );
            if ( n_update != states.n_cols ) {
                PetscPrintf( comm_,
                             "FSS: update_state_indices_status: states and status arrays have incompatible dimensions.\n" );
            }

            if ( n_update > 0 ) {
                for ( int i = 0; i < n_update; ++i ) {
                    local_dd_gids.col( i ) = arma::conv_to< arma::Col< ZOLTAN_ID_TYPE>>::from( states.col( i ));
                    lids( i ) = ( ZOLTAN_ID_TYPE ) local_ids( i );
                }
            } else {
                lids.set_size( 0 );
            }

            zoltan_err = Zoltan_DD_Update( state_directory_, local_dd_gids.memptr( ), &lids[ 0 ], &status[ 0 ], nullptr,
                                           n_update );
            ZOLTANCHKERRABORT( comm_, zoltan_err );
            logger_.event_end( logger_.total_update_dd_event );
        }

        void StateSetBase::retrieve_state_status( ) {
            int zoltan_err;

            local_states_status_.set_size( num_local_states_ );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE>>::from( local_states_ );

            zoltan_err = Zoltan_DD_Find( state_directory_, gids.colptr( 0 ), nullptr, local_states_status_.memptr( ),
                                         nullptr, num_local_states_, nullptr );
            ZOLTANCHKERRABORT( comm_, zoltan_err );
        }

        int StateSetBase::zoltan_num_frontier( void *data, int *ierr ) {
            *ierr = ZOLTAN_OK;
            return ( int ) (( StateSetBase * ) data )->frontiers_.n_cols;
        }

        int StateSetBase::zoltan_frontier_size( void *data, int num_gid_entries, int num_lid_entries,
                                                ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr ) {
            return (( StateSetBase * ) data )->num_species_ * sizeof( int );
        }

        void StateSetBase::zoltan_frontier_list( void *data, int num_gid_entries, int num_lid_entries,
                                                 ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim,
                                                 float *obj_wgts, int *ierr ) {
            auto my_data = ( StateSetBase * ) data;
            int n_frontier = ( int ) ( my_data->frontiers_ ).n_cols;

            for ( int i{0}; i < n_frontier; ++i ) {
                local_ids[ i ] = ( ZOLTAN_ID_TYPE ) i;
                global_id[ i ] = ( ZOLTAN_ID_TYPE ) my_data->local_frontier_gids_( i );
            }
            if ( wgt_dim == 1 ) {
                for ( int i{0}; i < n_frontier; ++i ) {
                    obj_wgts[ i ] = 1;
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void StateSetBase::pack_frontiers( void *data, int num_gid_entries, int num_lid_entries, int num_ids,
                                           ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *dest, int *sizes,
                                           int *idx, char *buf, int *ierr ) {
            *ierr = ZOLTAN_FATAL;
            auto my_data = ( StateSetBase * ) data;
            for ( int i{0}; i < num_ids; ++i ) {
                auto ptr = ( int * ) &buf[ idx[ i ]];
                for ( int j{0}; j < my_data->num_species_; ++j ) {
                    *( ptr + j ) = my_data->frontiers_( j, local_ids[ i ] );
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void StateSetBase::mid_frontier_migration( void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                                   ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                                   int *import_procs, int *import_to_part, int num_export,
                                                   ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
                                                   int *export_procs, int *export_to_part, int *ierr ) {
            auto my_data = ( StateSetBase * ) data;
            // remove the packed states from local data structure
            arma::uvec i_keep( my_data->frontier_lids_.n_elem );
            i_keep.zeros( );
            for ( int i{0}; i < num_export; ++i ) {
                i_keep( export_local_ids[ i ] ) = 1;
            }
            i_keep = arma::find( i_keep == 0 );

            my_data->frontiers_ = my_data->frontiers_.cols( i_keep );
        }

        void StateSetBase::unpack_frontiers( void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                                             int *sizes, int *idx, char *buf, int *ierr ) {

            auto my_data = ( StateSetBase * ) data;
            int nfrontier_old = my_data->frontiers_.n_cols;
            // Expand the data arrays
            my_data->frontiers_.resize( my_data->num_species_, nfrontier_old + num_ids );

            // Unpack new local states
            for ( int i{0}; i < num_ids; ++i ) {
                auto ptr = ( int * ) &buf[ idx[ i ]];
                for ( int j{0}; j < my_data->num_species_; ++j ) {
                    my_data->frontiers_( j, nfrontier_old + i ) = *( ptr + j );
                }
            }
        }

        void FiniteStateSubsetLogger::register_all( MPI_Comm comm ) {
            PetscErrorCode ierr;
            /// Register event logging
            ierr = PetscLogEventRegister( "State space exploration", 0, &state_exploration_event );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Check constraints", 0, &check_constraints_event );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Zoltan partitioning", 0, &call_partitioner_event );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Add states", 0, &add_states_event );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Zoltan DD stuff", 0, &zoltan_dd_stuff_event );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "update_state_indices()", 0, &total_update_dd_event );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "DistributeFrontiers()", 0, &distribute_frontiers_event );
            CHKERRABORT( comm, ierr );
        }

        void FiniteStateSubsetLogger::event_begin( PetscLogEvent event ) {
            PetscLogEventBegin( event, 0, 0, 0, 0 );
        }

        void FiniteStateSubsetLogger::event_end( PetscLogEvent event ) {
            PetscLogEventBegin( event, 0, 0, 0, 0 );
        }
    }
}
