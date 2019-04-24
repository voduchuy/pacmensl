//
// Created by Huy Vo on 12/4/18.
//
#include <FSS/FiniteStateSubset.h>
#include "FiniteStateSubset.h"


namespace cme {
    namespace parallel {
        FiniteStateSubset::FiniteStateSubset( MPI_Comm new_comm, PetscInt num_species ) {
            PetscErrorCode ierr;
            MPI_Comm_dup( new_comm, &comm );
            MPI_Comm_size( comm, &comm_size );
            MPI_Comm_rank( comm, &my_rank );
            n_species = num_species;
            max_num_molecules.set_size( n_species );
            local_states.resize( n_species, 0 );
            nstate_global = 0;
            stoichiometry.resize( n_species, 0 );

            /// Set up Zoltan load-balancing objects
            zoltan_lb = Zoltan_Create( comm );
            zoltan_explore = Zoltan_Create( comm );

            /// Set up Zoltan's parallel directory
            ierr = Zoltan_DD_Create( &state_directory, comm, n_species, 1, 1, hash_table_length, 0 );

            /// Register event logging
            ierr = PetscLogEventRegister( "State space exploration", 0, &state_exploration );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Check constraints", 0, &check_constraints );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Generate graph data", 0, &generate_graph_data );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Zoltan partitioning", 0, &call_partitioner );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Add states", 0, &add_states );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "Zoltan DD stuff", 0, &zoltan_dd_stuff );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "UpdateStateLID()", 0, &total_update_dd );
            CHKERRABORT( comm, ierr );
            ierr = PetscLogEventRegister( "DistributeFrontiers()", 0, &distribute_frontiers );
            CHKERRABORT( comm, ierr );
            /// Set up the default FSP hyper-rectangular shape
            rhs_constr.resize( n_species );
            lhs_constr = default_constr_fun;


            /// Layout of the mapping between states and Petsc ordering
            ierr = PetscLayoutCreate( comm, &state_layout );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutCreate( comm, &state_layout_no_sinks );
            CHKERRABORT( comm, ierr );
        };

        void FiniteStateSubset::SetStoichiometry( arma::Mat< PetscInt > SM ) {
            stoichiometry = SM;
            n_species = SM.n_rows;
            n_reactions = SM.n_cols;
            stoich_set = 1;
        }

        void FiniteStateSubset::SetLBType( PartitioningType lb_type ) {
            partitioning_type = lb_type;
        }

        void FiniteStateSubset::SetRepartApproach( PartitioningApproach approach ) {
            repart_approach = approach;
            zoltan_repart_opt = partapproach2str( repart_approach );
        }

        void FiniteStateSubset::SetShape(
                fsp_constr_multi_fn *lhs_fun,
                arma::Row< int > &rhs_bounds ) {
            lhs_constr = lhs_fun;
            rhs_constr = rhs_bounds;
        }


        void FiniteStateSubset::SetShapeBounds( arma::Row< PetscInt > &rhs_bounds ) {
            rhs_constr = arma::conv_to< arma::Row< int >>::from( rhs_bounds );
        }

        void FiniteStateSubset::SetInitialStates( arma::Mat< PetscInt > X0 ) {
            PetscInt my_rank;
            MPI_Comm_rank( comm, &my_rank );

            if ( X0.n_rows != n_species ) {
                throw std::runtime_error(
                        "SetInitialStates: number of rows in input array is not the same as the number of species.\n" );
            }

            PetscPrintf( comm, "Adding initial states...\n" );
            AddStates( X0 );
            PetscPrintf( comm, "Initial states set...\n" );
        }

        arma::Mat< PetscInt > FiniteStateSubset::GetLocalStates( ) {
            arma::Mat< PetscInt > states_return( n_species, nstate_local );
            for ( auto j{0}; j < nstate_local; ++j ) {
                for ( auto i{0}; i < n_species; ++i ) {
                    states_return( i, j ) = ( PetscInt ) local_states( i, j );
                }
            }
            return states_return;
        }

        FiniteStateSubset::~FiniteStateSubset( ) {
            PetscMPIInt ierr;
            ierr = MPI_Comm_free( &comm );
            MPICHKERRABORT( comm, ierr );
            Zoltan_Destroy( &zoltan_lb );
            Zoltan_DD_Destroy( &state_directory );
            ierr = PetscLayoutDestroy( &state_layout );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutDestroy( &state_layout_no_sinks );
            CHKERRABORT( comm, ierr );
        }

        arma::Row< PetscReal > FiniteStateSubset::SinkStatesReduce( Vec P ) {
            PetscInt ierr;

            arma::Row< PetscReal > local_sinks( rhs_constr.n_elem ), global_sinks( rhs_constr.n_elem );

            PetscInt p_local_size;
            ierr = VecGetLocalSize( P, &p_local_size );
            CHKERRABORT( comm, ierr );

            if ( p_local_size != local_states.n_cols + rhs_constr.n_elem ) {
                printf( "FiniteStateSubset::SinkStatesReduce: The layout of p and FiniteStateSubset do not match.\n" );
                MPI_Abort( comm, 1 );
            }

            PetscReal *p_data;
            VecGetArray( P, &p_data );
            for ( auto i{0}; i < rhs_constr.n_elem; ++i ) {
                local_sinks( i ) = p_data[ nstate_local + i ];
                ierr = MPI_Allreduce( &local_sinks[ i ], &global_sinks[ i ], 1, MPIU_REAL, MPI_SUM, comm );
                CHKERRABORT( comm, ierr );
            }

            return global_sinks;
        }

        arma::Col< PetscReal > FiniteStateSubset::marginal( Vec P, PetscInt species ) {
            PetscReal *local_data;
            VecGetArray( P, &local_data );

            arma::Col< PetscReal > p_local( local_data, nstate_local, false, true );
            arma::Col< PetscReal > v( max_num_molecules( species ) + 1 );
            v.fill( 0.0 );

            for ( PetscInt i{0}; i < nstate_local; ++i ) {
                v( local_states( species, i )) += p_local( i );
            }

            MPI_Barrier( comm );

            arma::Col< PetscReal > w( max_num_molecules( species ) + 1 );
            w.fill( 0.0 );
            MPI_Allreduce( &v[ 0 ], &w[ 0 ], v.size( ), MPI_DOUBLE, MPI_SUM, comm );

            VecRestoreArray( P, &local_data );
            return w;
        }

        PetscInt FiniteStateSubset::GetNumGlobalStates( ) {
            return nstate_global;
        }

        void FiniteStateSubset::GenerateStatesAndOrdering( ) {
            InitZoltanParameters( );

            bool frontier_empty;

            RetrieveStateStatus( );
            frontier_lids = arma::find( local_states_status == 1 );

            PetscLogEventBegin( state_exploration, 0, 0, 0, 0 );
            // Check if the set of frontier states are empty on all processors
            {
                int n1, n2;
                n1 = ( int ) frontier_lids.n_elem;
                MPI_Allreduce( &n1, &n2, 1, MPI_INT, MPI_MAX, comm );
                frontier_empty = ( n2 == 0 );
            }

            arma::Row< char > frontier_status;
            while ( !frontier_empty ) {
                // Distribute frontier states to all processors
                frontiers = local_states.cols( frontier_lids );
                DistributeFrontier( );
                frontier_status.set_size( frontiers.n_cols );
                frontier_status.fill( 0 );

                arma::Mat< PetscInt > Y( n_species, frontiers.n_cols * n_reactions );
                arma::Row< PetscInt > ystatus( Y.n_cols );

                for ( int i{0}; i < frontiers.n_cols; i++ ) {
                    for ( int j{0}; j < n_reactions; ++j ) {
                        Y.col( j * frontiers.n_cols + i ) = frontiers.col( i ) + stoichiometry.col( j );
                        ystatus( j * frontiers.n_cols + i ) = CheckState( Y.colptr( j * frontiers.n_cols + i ));

                        if ( ystatus( j * frontiers.n_cols + i ) < 0 ) {
                            frontier_status( i ) = -1;
                        }
                    }
                }
                Y = Y.cols( arma::find( ystatus == 0 ));
                Y = unique_columns( Y );
                AddStates( Y );

                // Deactivate states whose neighbors have all been explored and added to the state set
                UpdateStateStatus( frontiers, frontier_status );
                RetrieveStateStatus( );
                frontier_lids = arma::find( local_states_status == 1 );

                // Check if the set of frontier states are empty on all processors
                {
                    int n1, n2;
                    n1 = ( int ) frontier_lids.n_elem;
                    MPI_Allreduce( &n1, &n2, 1, MPI_INT, MPI_MAX, comm );
                    frontier_empty = ( n2 == 0 );
                }
            }
            PetscLogEventEnd( state_exploration, 0, 0, 0, 0 );

            PetscLogEventBegin( call_partitioner, 0, 0, 0, 0 );
            if (comm_size > 1){
                // Repartition the state set
                if ( partitioning_type == Graph ) {
                    PetscMPIInt n_local_min;
                    PetscMPIInt n_local = ( PetscMPIInt ) nstate_local;
                    MPI_Allreduce( &n_local, &n_local_min, 1, MPI_INT, MPI_MIN, comm );
                    if ( n_local_min == 0 ) {
                        Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "Block" );
                    }
                }

                if ( nstate_global_old * ( 1.0 + lb_threshold ) <= 1.0 * nstate_global || nstate_global_old == 0 ) {
                    nstate_global_old = nstate_global;
                    PetscPrintf( comm, "Repartitioning...\n" );
                    LoadBalance( );
                }
            }
            PetscLogEventEnd( call_partitioner, 0, 0, 0, 0 );

            // Switch states with status -1 to 1, for they may expand to new states when the shape constraints are relaxed
            RetrieveStateStatus( );
            arma::uvec iupdate = arma::find( local_states_status == -1 );
            arma::Mat< int > states_update;
            arma::Row< char > new_status;
            if ( iupdate.n_elem > 0 ) {
                new_status.set_size( iupdate.n_elem );
                new_status.fill( 1 );
                states_update = local_states.cols( iupdate );
            } else {
                states_update.set_size( n_species, 0 );
                new_status.set_size( 0 );
            }
            UpdateStateStatus( states_update, new_status );
        }

        void FiniteStateSubset::DistributeFrontier( ) {
            PetscLogEventBegin( distribute_frontiers, 0, 0, 0, 0 );
            UpdateMaxNumMolecules( );

            local_frontier_gids = sub2ind_nd( max_num_molecules, frontiers );

            // Variables to store Zoltan's output
            int zoltan_err, ierr;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            zoltan_err = Zoltan_LB_Partition( zoltan_explore, &changes, &num_gid_entries, &num_lid_entries, &num_import,
                                              &import_global_ids, &import_local_ids,
                                              &import_procs, &import_to_part, &num_export, &export_global_ids,
                                              &export_local_ids, &export_procs, &export_to_part );
            ZOLTANCHKERRABORT( comm, zoltan_err );

            Zoltan_LB_Free_Part( &import_global_ids, &import_local_ids, &import_procs, &import_to_part );
            Zoltan_LB_Free_Part( &export_global_ids, &export_local_ids, &export_procs, &export_to_part );
            PetscLogEventEnd( distribute_frontiers, 0, 0, 0, 0 );
        }

        void FiniteStateSubset::LoadBalance( ) {
            PetscLogEventBegin( generate_graph_data, 0, 0, 0, 0 );
            GenerateGraphData( );
            PetscLogEventEnd( generate_graph_data, 0, 0, 0, 0 );

            // Variables to store Zoltan's output
            int zoltan_err, ierr;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            zoltan_err = Zoltan_LB_Partition( zoltan_lb, &changes, &num_gid_entries, &num_lid_entries, &num_import,
                                              &import_global_ids, &import_local_ids,
                                              &import_procs, &import_to_part, &num_export, &export_global_ids,
                                              &export_local_ids, &export_procs, &export_to_part );
            ZOLTANCHKERRABORT( comm, zoltan_err );

            nstate_local = local_states.n_cols;
            PetscMPIInt nslocal = nstate_local;
            PetscMPIInt nsglobal;
            MPI_Allreduce( &nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm );
            nstate_global = nsglobal;

            UpdateLayouts( );
            UpdateStateLID( );

            Zoltan_LB_Free_Part( &import_global_ids, &import_local_ids, &import_procs, &import_to_part );
            Zoltan_LB_Free_Part( &export_global_ids, &export_local_ids, &export_procs, &export_to_part );
        }

        void FiniteStateSubset::AddStates( arma::Mat< PetscInt > &X ) {
            PetscLogEventBegin( add_states, 0, 0, 0, 0 );
            int zoltan_err;

            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids; //
            arma::Row< int > owner( X.n_cols );
            arma::Row< int > parts( X.n_cols );
            arma::uvec iselect;


            // Probe if states in X are already owned by some processor
            local_dd_gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE>>::from( X );
            zoltan_err = Zoltan_DD_Find( state_directory, &local_dd_gids[ 0 ], nullptr, nullptr, nullptr, X.n_cols,
                                         &owner[ 0 ] );
            iselect = arma::find( owner == -1 );
            ZOLTANCHKERRABORT( comm, zoltan_err );

            // Shed any states that are already included
            X = X.cols( iselect );
            local_dd_gids = local_dd_gids.cols( iselect );

            // Add local states to Zoltan directory (only 1 state from 1 processor will be added in case of overlapping)
            parts.resize( X.n_cols );
            parts.fill( my_rank );
            PetscLogEventBegin( zoltan_dd_stuff, 0, 0, 0, 0 );
            zoltan_err = Zoltan_DD_Update( state_directory, &local_dd_gids[ 0 ], nullptr, nullptr, parts.memptr( ),
                                           X.n_cols );
            ZOLTANCHKERRABORT( comm, zoltan_err );
            PetscLogEventEnd( zoltan_dd_stuff, 0, 0, 0, 0 );

            // Remove overlaps between processors
            parts.resize( X.n_cols );
            zoltan_err = Zoltan_DD_Find( state_directory, &local_dd_gids[ 0 ], nullptr, nullptr, parts.memptr( ),
                                         X.n_cols,
                                         nullptr );
            ZOLTANCHKERRABORT( comm, zoltan_err );

            iselect = arma::find( parts == my_rank );
            X = X.cols( iselect );

            arma::Row< char > X_status( X.n_cols );
            if ( X.n_cols > 0 ) {
                // Append X to local states
                X_status.fill( 1 );
                local_states = arma::join_horiz( local_states, X );
            }

            PetscInt nstate_local_old = nstate_local;
            nstate_local = ( PetscInt ) local_states.n_cols;
            PetscMPIInt nslocal = nstate_local;
            PetscMPIInt nsglobal;
            MPI_Allreduce( &nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm );
            nstate_global = nsglobal;

            // Update layout
            UpdateLayouts( );

            // Update local ids and status
            arma::Row< PetscInt > local_ids;
            if ( nstate_local > nstate_local_old ) {
                local_ids = arma::regspace< arma::Row< PetscInt>>( nstate_local_old, nstate_local - 1 );
            } else {
                local_ids.resize( 0 );
            }

            UpdateStateLIDStatus( X, local_ids, X_status );
            PetscLogEventEnd( add_states, 0, 0, 0, 0 );
        }

        void FiniteStateSubset::UpdateMaxNumMolecules( ) {
            int ierr;
            arma::Col< PetscInt > local_max_num_molecules( n_species );

            if ( nstate_local > 0 ) {
                local_max_num_molecules = arma::max( local_states, 1 );

                for ( int ir{0}; ir < n_reactions; ++ir ) {
                    for ( int i{0}; i < n_species; ++i ) {
                        local_max_num_molecules( i ) = std::max( local_max_num_molecules( i ),
                                                                 local_max_num_molecules( i ) + stoichiometry( i, ir ));
                        local_max_num_molecules( i ) = std::max( local_max_num_molecules( i ),
                                                                 local_max_num_molecules( i ) - stoichiometry( i, ir ));
                    }
                }
            } else {
                local_max_num_molecules.zeros( );
            }

            ierr = MPI_Allreduce( local_max_num_molecules.memptr( ), max_num_molecules.memptr( ), n_species, MPI_INT,
                                  MPI_MAX, comm );
            MPICHKERRABORT( comm, ierr );
        }

        PetscInt FiniteStateSubset::CheckState( PetscInt *x ) {
            for ( int i1{0}; i1 < n_species; ++i1 ) {
                if ( x[ i1 ] < 0 ) {
                    return -1;
                }
            }
            int *fval;
            fval = new int[rhs_constr.n_elem];
            lhs_constr( n_species, rhs_constr.n_elem, 1, x, &fval[ 0 ] );

            for ( int i{0}; i < rhs_constr.n_elem; ++i ) {
                if ( fval[ i ] > rhs_constr( i )) {
                    return -1;
                }
            }
            delete[] fval;
            return 0;
        }

        void
        FiniteStateSubset::CheckConstraint( PetscInt num_states, PetscInt *x, PetscInt *satisfied ) {
            auto *fval = new int[num_states * rhs_constr.n_elem];
            lhs_constr( n_species, rhs_constr.n_elem, num_states, x, fval );
            for ( int iconstr = 0; iconstr < rhs_constr.n_elem; ++iconstr ) {
                for ( int i = 0; i < num_states; ++i ) {
                    satisfied[ num_states * iconstr + i ] = ( fval[ rhs_constr.n_elem * i + iconstr ] <=
                                                              rhs_constr( iconstr )) ? 1 : 0;
                    for ( int j = 0; j < n_species; ++j ) {
                        if ( x[ n_species * i + j ] < 0 ) {
                            satisfied[ num_states * iconstr + i ] = 1;
                        }
                    }
                }
            }
            delete[] fval;
        }

        void FiniteStateSubset::GenerateGraphData( ) {

            UpdateMaxNumMolecules( );
            local_graph_gids = State2Petsc( local_states, false );

            local_observable_states.set_size( nstate_local, n_reactions );
            num_local_edges.set_size( nstate_local );
            num_local_edges.fill( 0 );
            state_weights.set_size( nstate_local );
            state_weights.fill( 4.0f * n_reactions );
            arma::Mat< PetscInt > reachable_states( n_species, nstate_local );
            for ( int ir{0}; ir < n_reactions; ++ir ) {
                reachable_states = local_states - arma::repmat( stoichiometry.col( ir ), 1, nstate_local );
                State2Petsc( reachable_states, local_observable_states.colptr( ir ), false );
                for ( int i{0}; i < nstate_local; ++i ) {
                    if ( local_observable_states( i, ir ) >= 0 ) {
                        num_local_edges( i ) += 1;
                        state_weights( i ) += 2.0f;
                    }
                }
            }

            if ( partitioning_type != Graph ) {
                return;
            }

            local_reachable_states.set_size( nstate_local, n_reactions );
            for ( int ir{0}; ir < n_reactions; ++ir ) {
                reachable_states = local_states + repmat( stoichiometry.col( ir ), 1, nstate_local );
                State2Petsc( reachable_states, local_reachable_states.colptr( ir ), false );
                for ( int i{0}; i < nstate_local; ++i ) {
                    if ( local_reachable_states( i, ir ) >= 0 ) {
                        num_local_edges( i ) += 1;
                        state_weights( i ) += 2.0f;
                    }
                }
            }
        }

        void FiniteStateSubset::InitZoltanParameters( ) {
            // Parameters for state exploration load-balancing
            Zoltan_Set_Param( zoltan_explore, "NUM_GID_ENTRIES", "1" );
            Zoltan_Set_Param( zoltan_explore, "NUM_LID_ENTRIES", "1" );
            Zoltan_Set_Param( zoltan_explore, "IMBALANCE_TOL", "1.01" );
            Zoltan_Set_Param( zoltan_explore, "AUTO_MIGRATE", "1" );
            Zoltan_Set_Param( zoltan_explore, "RETURN_LISTS", "ALL" );
            Zoltan_Set_Param( zoltan_explore, "DEBUG_LEVEL", "0" );
            Zoltan_Set_Param( zoltan_explore, "LB_METHOD", "Block" );
            Zoltan_Set_Num_Obj_Fn( zoltan_explore, &zoltan_num_frontier, ( void * ) this );
            Zoltan_Set_Obj_List_Fn( zoltan_explore, &zoltan_frontier_list, ( void * ) this );
            Zoltan_Set_Obj_Size_Fn( zoltan_explore, &zoltan_obj_size, ( void * ) this );
            Zoltan_Set_Pack_Obj_Multi_Fn( zoltan_explore, &zoltan_pack_frontiers, ( void * ) this );
            Zoltan_Set_Mid_Migrate_PP_Fn( zoltan_explore, &zoltan_frontiers_mid_migrate_pp, ( void * ) this );
            Zoltan_Set_Unpack_Obj_Multi_Fn( zoltan_explore, &zoltan_unpack_frontiers, ( void * ) this );

            // Parameters for computational load-balancing
            Zoltan_Set_Param( zoltan_lb, "NUM_GID_ENTRIES", "1" );
            Zoltan_Set_Param( zoltan_lb, "NUM_LID_ENTRIES", "1" );
            Zoltan_Set_Param( zoltan_lb, "AUTO_MIGRATE", "1" );
            Zoltan_Set_Param( zoltan_lb, "RETURN_LISTS", "ALL" );
            Zoltan_Set_Param( zoltan_lb, "DEBUG_LEVEL", "0" );
            // This imbalance tolerance is universal for all methods
            Zoltan_Set_Param( zoltan_lb, "IMBALANCE_TOL", "1.01" );
            Zoltan_Set_Param( zoltan_lb, "LB_APPROACH", zoltan_repart_opt.c_str( ));
            Zoltan_Set_Param( zoltan_lb, "GRAPH_BUILD_TYPE", "FAST_NO_DUP" );
            // Register query functions to zoltan_lb
            Zoltan_Set_Num_Obj_Fn( zoltan_lb, &zoltan_num_obj, ( void * ) this );
            Zoltan_Set_Obj_List_Fn( zoltan_lb, &zoltan_obj_list, ( void * ) this );
            Zoltan_Set_Obj_Size_Fn( zoltan_lb, &zoltan_obj_size, ( void * ) this );
            Zoltan_Set_Pack_Obj_Multi_Fn( zoltan_lb, &zoltan_pack_states, ( void * ) this );
            Zoltan_Set_Unpack_Obj_Multi_Fn( zoltan_lb, &zoltan_unpack_states, ( void * ) this );
            Zoltan_Set_Mid_Migrate_PP_Fn( zoltan_lb, &zoltan_mid_migrate_pp, ( void * ) this );
            Zoltan_Set_Num_Edges_Fn( zoltan_lb, &zoltan_num_edges, ( void * ) this );
            Zoltan_Set_Edge_List_Multi_Fn( zoltan_lb, &zoltan_get_graph_edges, ( void * ) this );
            Zoltan_Set_HG_Size_CS_Fn( zoltan_lb, &zoltan_get_hypergraph_size, ( void * ) this );
            Zoltan_Set_HG_CS_Fn( zoltan_lb, &zoltan_get_hypergraph, ( void * ) this );

            switch ( partitioning_type ) {
                case Block:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "Block" );
                    break;
                case Graph:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "GRAPH" );
                    Zoltan_Set_Param( zoltan_lb, "GRAPH_PACKAGE", "Parmetis" );
                    Zoltan_Set_Param( zoltan_lb, "OBJ_WEIGHT_DIM", "1" );
                    Zoltan_Set_Param( zoltan_lb, "EDGE_WEIGHT_DIM", "0" );
                    Zoltan_Set_Param( zoltan_lb, "CHECK_GRAPH", "0" );
                    Zoltan_Set_Param( zoltan_lb, "GRAPH_SYMMETRIZE", "NONE" );
                    Zoltan_Set_Param( zoltan_lb, "PARMETIS_ITR", "100" );
                    break;
                case HyperGraph:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "HYPERGRAPH" );
                    Zoltan_Set_Param( zoltan_lb, "HYPERGRAPH_PACKAGE", "PHG" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_CUT_OBJECTIVE", "CONNECTIVITY" );
                    Zoltan_Set_Param( zoltan_lb, "CHECK_HYPERGRAPH", "0" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_REPART_MULTIPLIER", "100" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_RANDOMIZE_INPUT", "1" );
                    Zoltan_Set_Param( zoltan_lb, "OBJ_WEIGHT_DIM", "1" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_COARSENING_METHOD", "AGG" );
                    Zoltan_Set_Param( zoltan_lb, "EDGE_WEIGHT_DIM", "0" );// use Zoltan default hyperedge weights
                    Zoltan_Set_Param( zoltan_lb, "PHG_EDGE_WEIGHT_OPERATION", "ADD" );
                    break;
            }
        }

        arma::Row< PetscInt > FiniteStateSubset::State2Petsc( arma::Mat< PetscInt > state, bool count_sinks ) {
            arma::Row< PetscInt > indices( state.n_cols );
            indices.fill( 0 );

            for ( int i{0}; i < state.n_cols; ++i ) {
                for ( int j = 0; j < n_species; ++j ) {
                    if ( state( j, i ) < 0 ) {
                        indices( i ) = -1;
                        break;
                    }
                }
            }

            arma::uvec i_vaild_constr = arma::find( indices == 0 );
            state = state.cols( i_vaild_constr );

            arma::Row< ZOLTAN_ID_TYPE > lids( state.n_cols );
            arma::Row< int > parts( state.n_cols );
            arma::Row< int > owners( state.n_cols );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE >>::from( state );

            Zoltan_DD_Find( state_directory, gids.memptr( ), lids.memptr( ), nullptr, parts.memptr( ), state.n_cols,
                            owners.memptr() );

            const PetscInt *starts;
            if ( count_sinks ) {
                PetscLayoutGetRanges( state_layout, &starts );
            } else {
                PetscLayoutGetRanges( state_layout_no_sinks, &starts );
            }


            for ( int i{0}; i < state.n_cols; i++ ) {
                int indx = i_vaild_constr( i );
                if ( owners[i] >= 0 && parts[ i ] >= 0 ) {
                    indices( indx ) = starts[ parts[ i ]] + ( PetscInt ) lids( i );
                } else {
                    indices( indx ) = -1;
                }
            }

            return indices;
        }

        void FiniteStateSubset::State2Petsc( arma::Mat< PetscInt > state, PetscInt *indx, bool count_sinks ) {

            arma::Row< PetscInt > ipositive( state.n_cols );
            ipositive.fill( 0 );

            for ( int i{0}; i < ipositive.n_cols; ++i ) {
                for ( int j = 0; j < n_species; ++j ) {
                    if ( state( j, i ) < 0 ) {
                        ipositive( i ) = -1;
                        indx[ i ] = -1;
                        break;
                    }
                }
            }

            arma::uvec i_vaild_constr = arma::find( ipositive == 0 );
            state = state.cols( i_vaild_constr );

            arma::Row< ZOLTAN_ID_TYPE > lids( state.n_cols );
            arma::Row< int > parts( state.n_cols );
            arma::Row< int > owners( state.n_cols );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE >>::from( state );

            Zoltan_DD_Find( state_directory, gids.memptr( ), lids.memptr( ), nullptr, parts.memptr( ), state.n_cols,
                            owners.memptr() );

            const PetscInt *starts;
            if ( count_sinks ) {
                PetscLayoutGetRanges( state_layout, &starts );
            } else {
                PetscLayoutGetRanges( state_layout_no_sinks, &starts );
            }
            for ( int i{0}; i < state.n_cols; i++ ) {
                auto ii = i_vaild_constr( i );
                if ( owners[i] >= 0 && parts[ i ] >= 0 ) {
                    indx[ ii ] = starts[ parts[ i ]] + ( PetscInt ) lids( i );
                } else {
                    indx[ ii ] = -1;
                }
            }
        }

        /*
         * Getters
         */

        std::tuple< PetscInt, PetscInt > FiniteStateSubset::GetLayoutStartEnd( ) {
            PetscInt start, end, ierr;
            ierr = PetscLayoutGetRange( state_layout, &start, &end );
            CHKERRABORT( comm, ierr );
            return std::make_tuple( start, end );
        }


        MPI_Comm FiniteStateSubset::GetComm( ) {
            return comm;
        }

        PetscInt FiniteStateSubset::GetNumLocalStates( ) {
            return nstate_local;
        }

        PetscInt FiniteStateSubset::GetNumSpecies( ) {
            return ( PetscInt( n_species ));
        }

        PetscInt FiniteStateSubset::GetNumReactions( ) {
            return stoichiometry.n_cols;
        }

        arma::Row< int > FiniteStateSubset::GetShapeBounds( ) {
            return arma::Row< int >( rhs_constr );
        }

        PetscInt FiniteStateSubset::GetNumConstraints( ) {
            return rhs_constr.n_elem;
        }

        void FiniteStateSubset::default_constr_fun( int num_species, int num_constr, int n_states, int *states,
                                                    int *outputs ) {
            for ( int i{0}; i < n_states * num_species; ++i ) {
                outputs[ i ] = states[ i ];
            }
        }

        void FiniteStateSubset::GiveZoltanObjList( int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids,
                                                   ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr ) {

            for ( int i{0}; i < nstate_local; ++i ) {
                global_ids[ i ] = ( ZOLTAN_ID_TYPE ) local_graph_gids( i );
                local_ids[ i ] = ( ZOLTAN_ID_TYPE ) i;
            }
            if ( wgt_dim == 1 ) {
                for ( int i{0}; i < nstate_local; ++i ) {
                    obj_wgts[ i ] = state_weights( i );
                }
            }
        }

        int FiniteStateSubset::GiveZoltanObjSize( int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                                  ZOLTAN_ID_PTR local_id, int *ierr ) {
            return ( int ) sizeof( PetscInt );
        }

        void
        FiniteStateSubset::GiveZoltanSendBuffer( int num_gid_entries, int num_lid_entries, int num_ids,
                                                 ZOLTAN_ID_PTR global_ids,
                                                 ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf,
                                                 int *ierr ) {
            for ( int i{0}; i < num_ids; ++i ) {
                auto ptr = ( PetscInt * ) &buf[ idx[ i ]];
                auto state_id = local_ids[ i ];
                // pack the state's lexicographic index
                sub2ind_nd( n_species, &max_num_molecules[ 0 ], 1, local_states.colptr( state_id ), ptr );
            }
        }

        void FiniteStateSubset::MidMigrationProcessing( int num_gid_entries, int num_lid_entries, int num_import,
                                                        ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                                        int *import_procs, int *import_to_part, int num_export,
                                                        ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
                                                        int *export_procs, int *export_to_part, int *ierr ) {
            // remove the packed states from local data structure
            arma::uvec i_keep( nstate_local );
            i_keep.zeros( );
            for ( int i{0}; i < num_export; ++i ) {
                i_keep( export_local_ids[ i ] ) = 1;
            }
            i_keep = arma::find( i_keep == 0 );

            local_states = local_states.cols( i_keep );
            nstate_local = ( PetscInt ) local_states.n_cols;

            // Expand the data arrays
            local_states.resize( n_species, nstate_local + num_import );
        }

        void
        FiniteStateSubset::ReceiveZoltanBuffer( int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                                                int *sizes, int *idx, char *buf, int *ierr ) {

            // Unpack new local states
            for ( int i{0}; i < num_ids; ++i ) {
                auto ptr = ( PetscInt * ) &buf[ idx[ i ]];
                ind2sub_nd< PetscInt, PetscInt >( n_species, &max_num_molecules[ 0 ], 1, ptr,
                                                  local_states.colptr( nstate_local + i ));
            }
            nstate_local = nstate_local + num_ids;
        }

        int FiniteStateSubset::GiveZoltanNumFrontier( ) {
            return ( int ) frontiers.n_cols;
        }

        void FiniteStateSubset::GiveZoltanFrontierList( int num_gid_entries, int num_lid_entries,
                                                        ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim,
                                                        float *obj_wgts, int *ierr ) {
            int n_frontier = ( int ) frontiers.n_cols;

            for ( int i{0}; i < n_frontier; ++i ) {
                local_ids[ i ] = ( ZOLTAN_ID_TYPE ) i;
                global_id[ i ] = ( ZOLTAN_ID_TYPE ) local_frontier_gids( i );
            }
            if ( wgt_dim == 1 ) {
                for ( int i{0}; i < n_frontier; ++i ) {
                    obj_wgts[ i ] = 1;
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void
        FiniteStateSubset::PackFrontiers( int num_gid_entries, int num_lid_entries, int num_ids,
                                          ZOLTAN_ID_PTR global_ids,
                                          ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf,
                                          int *ierr ) {
            for ( int i{0}; i < num_ids; ++i ) {
                auto ptr = ( PetscInt * ) &buf[ idx[ i ]];
                *ptr = local_frontier_gids( local_ids[i] );
            }
        }

        void FiniteStateSubset::FrontiersMidMigration( int num_gid_entries, int num_lid_entries, int num_import,
                                                       ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                                       int *import_procs, int *import_to_part, int num_export,
                                                       ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
                                                       int *export_procs, int *export_to_part, int *ierr ) {
            // remove the packed states from local data structure
            arma::uvec i_keep( frontier_lids.n_elem );
            i_keep.zeros( );
            for ( int i{0}; i < num_export; ++i ) {
                i_keep( export_local_ids[ i ] ) = 1;
            }
            i_keep = arma::find( i_keep == 0 );

            frontiers = frontiers.cols( i_keep );
        }

        void
        FiniteStateSubset::ReceiveFrontiers( int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                                             int *sizes, int *idx, char *buf, int *ierr ) {

            int nfrontier_old = frontiers.n_cols;
            // Expand the data arrays
            frontiers.resize( n_species, nfrontier_old + num_ids );

            // Unpack new local states
            for ( int i{0}; i < num_ids; ++i ) {
                auto ptr = ( PetscInt * ) &buf[ idx[ i ]];
                ind2sub_nd< PetscInt, PetscInt >( n_species, &max_num_molecules[ 0 ], 1, ptr,
                                                  frontiers.colptr( nfrontier_old + i ));
            }
        }

        int FiniteStateSubset::GiveZoltanNumEdges( int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                                   ZOLTAN_ID_PTR local_id, int *ierr ) {
            ZOLTAN_ID_TYPE indx = *local_id;
            *ierr = ZOLTAN_OK;
            return num_local_edges( indx );
        }

        void FiniteStateSubset::GiveZoltanGraphEdges( int num_gid_entries, int num_lid_entries, int num_obj,
                                                      ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *num_edges,
                                                      ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                                                      float *ewgts, int *ierr ) {

            if ( nstate_local != num_obj ) {
                *ierr = ZOLTAN_FATAL;
                return;
            }

            int nbr_ptr = 0;
            for ( int i{0}; i < nstate_local; ++i ) {
                global_id[ i ] = ( ZOLTAN_ID_TYPE ) local_graph_gids( i );
                local_id[ i ] = ( ZOLTAN_ID_TYPE ) i;
                num_edges[ i ] = num_local_edges[ i ];
                for ( int ir{0}; ir < n_reactions; ++ir ) {
                    if ( local_observable_states( i, ir ) >= 0 ) {
                        nbor_global_id[ nbr_ptr ] = ( ZOLTAN_ID_TYPE ) local_observable_states( i, ir );
                        nbr_ptr += 1;
                    }
                }
                if ( partitioning_type == Graph ) {
                    for ( int ir{0}; ir < n_reactions; ++ir ) {
                        if ( local_reachable_states( i, ir ) >= 0 ) {
                            nbor_global_id[ nbr_ptr ] = ( ZOLTAN_ID_TYPE ) local_reachable_states( i, ir );
                            nbr_ptr += 1;
                        }
                    }
                }
            }

            if ( wgt_dim == 1 ) {
                for ( int i{0}; i < nstate_local; ++i ) {
                    ewgts[ i ] = 1.0f * sizeof( PetscReal );
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubset::GiveZoltanHypergraphSize( int *num_lists, int *num_pins, int *format, int *ierr ) {
            *num_lists = nstate_local;
            *num_pins = ( int ) arma::sum( num_local_edges ) + nstate_local;
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubset::GiveZoltanHypergraph( int num_gid_entries, int num_vertices, int num_pins, int format,
                                                      ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr,
                                                      ZOLTAN_ID_PTR pin_gid, int *ierr ) {
            int pin_ptr{0};
            for ( int i{0}; i < num_vertices; ++i ) {
                if ( i == 0 ) {
                    vtx_edge_ptr[ 0 ] = 0;
                } else {
                    vtx_edge_ptr[ i ] = vtx_edge_ptr[ i - 1 ] + num_local_edges( i - 1 ) + 1;
                }

                vtx_gid[ i ] = local_graph_gids( i );

                pin_gid[ pin_ptr ] = local_graph_gids( i );

                pin_ptr += 1;

                for ( int ir{0}; ir < n_reactions; ++ir ) {
                    if ( local_observable_states( i, ir ) >= 0 ) {
                        pin_gid[ pin_ptr ] = ( ZOLTAN_ID_TYPE ) local_observable_states( i, ir );
                        pin_ptr += 1;
                    }
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubset::UpdateLayouts( ) {
            PetscErrorCode ierr;

            ierr = PetscLayoutDestroy( &state_layout );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutCreate( comm, &state_layout );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutSetLocalSize( state_layout, nstate_local + rhs_constr.n_elem );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutSetUp( state_layout );
            CHKERRABORT( comm, ierr );

            ierr = PetscLayoutDestroy( &state_layout_no_sinks );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutCreate( comm, &state_layout_no_sinks );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutSetLocalSize( state_layout_no_sinks, nstate_local );
            CHKERRABORT( comm, ierr );
            ierr = PetscLayoutSetUp( state_layout_no_sinks );
            CHKERRABORT( comm, ierr );
        }

        void FiniteStateSubset::UpdateStateLID( ) {
            PetscLogEventBegin( total_update_dd, 0, 0, 0, 0 );
            int zoltan_err;
            // Update hash table
            auto local_dd_gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE>>::from( local_states );
            arma::Row< ZOLTAN_ID_TYPE > lids;
            if ( nstate_local > 0 ) {
                lids = arma::regspace< arma::Row< ZOLTAN_ID_TYPE>>( 0, nstate_local - 1 );
            } else {
                lids.set_size( 0 );
            }
            arma::Row<int> parts(nstate_local);
            parts.fill(my_rank);
            zoltan_err = Zoltan_DD_Update( state_directory, &local_dd_gids[ 0 ], &lids[ 0 ], nullptr, parts.memptr(),
                                           nstate_local );
            ZOLTANCHKERRABORT( comm, zoltan_err );
            PetscLogEventEnd( total_update_dd, 0, 0, 0, 0 );
        }

        void FiniteStateSubset::UpdateStateStatus( arma::Mat< PetscInt > states, arma::Row< char > status ) {
            PetscLogEventBegin( total_update_dd, 0, 0, 0, 0 );
            int zoltan_err;

            // Update hash table
            PetscInt n_update = status.n_elem;
            if (n_update != states.n_cols){
                PetscPrintf(comm, "FSS: UpdateStateStatus: states and status arrays have incompatible dimensions.\n");
            }
            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids( n_species, n_update );

            if ( n_update > 0 ) {
                for ( PetscInt i = 0; i < n_update; ++i ) {
                    local_dd_gids.col( i ) = arma::conv_to< arma::Col< ZOLTAN_ID_TYPE>>::from(
                            states.col( i ));
                }
            }

            zoltan_err = Zoltan_DD_Update( state_directory, local_dd_gids.memptr(), nullptr, status.memptr( ), nullptr,
                                           n_update );
            ZOLTANCHKERRABORT( comm, zoltan_err );
            PetscLogEventEnd( total_update_dd, 0, 0, 0, 0 );
        }


        void FiniteStateSubset::UpdateStateLIDStatus( arma::Mat< PetscInt > states, arma::Row< PetscInt > local_ids,
                                                      arma::Row< char > status ) {
            PetscLogEventBegin( total_update_dd, 0, 0, 0, 0 );
            int zoltan_err;

            // Update hash table
            PetscInt n_update = local_ids.n_elem;
            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids( n_species, n_update );
            arma::Row< ZOLTAN_ID_TYPE > lids( n_update );
            if (n_update != states.n_cols){
                PetscPrintf(comm, "FSS: UpdateStateLIDStatus: states and status arrays have incompatible dimensions.\n");
            }

            if ( n_update > 0 ) {
                for ( PetscInt i = 0; i < n_update; ++i ) {
                    local_dd_gids.col( i ) = arma::conv_to< arma::Col< ZOLTAN_ID_TYPE>>::from(
                            states.col( i ));
                    lids( i ) = ( ZOLTAN_ID_TYPE ) local_ids( i );
                }
            } else {
                lids.set_size( 0 );
            }

            zoltan_err = Zoltan_DD_Update( state_directory, local_dd_gids.memptr(), &lids[ 0 ], &status[ 0 ], nullptr,
                                           n_update );
            ZOLTANCHKERRABORT( comm, zoltan_err );
            PetscLogEventEnd( total_update_dd, 0, 0, 0, 0 );
        }

        void FiniteStateSubset::RetrieveStateStatus( ) {
            int zoltan_err;

            local_states_status.set_size( nstate_local );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE>>::from( local_states );

            zoltan_err = Zoltan_DD_Find( state_directory, gids.colptr( 0 ), nullptr, local_states_status.memptr( ),
                                         nullptr, nstate_local, nullptr );
            ZOLTANCHKERRABORT( comm, zoltan_err );
        }
    }
}
