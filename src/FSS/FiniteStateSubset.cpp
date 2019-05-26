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
            Zoltan_Destroy( &zoltan_explore );
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
            if ( comm_size > 1 ) {
                // Repartition the state set
                if ( partitioning_type == Graph || partitioning_type == Hierarchical ) {
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
            if ( partitioning_type == Graph || partitioning_type == Hierarchical ) {
                GenerateGraphData( );
            } else {
                GenerateHyperGraphData( );
            }
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

            if ( partitioning_type == Graph || partitioning_type == Hierarchical ) {
                FreeGraphData( );
            } else {
                FreeHyperGraphData( );
            }
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
            Zoltan_Set_HG_Size_Edge_Wts_Fn( zoltan_lb, &zoltan_get_hg_size_eweights, ( void * ) this );
            Zoltan_Set_HG_Edge_Wts_Fn( zoltan_lb, &zoltan_get_hg_eweights, ( void * ) this );

            switch ( partitioning_type ) {
                case Block:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "Block" );
                    Zoltan_Set_Param( zoltan_lb, "OBJ_WEIGHT_DIM", "1" );
                    break;
                case Graph:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "GRAPH" );
                    Zoltan_Set_Param( zoltan_lb, "GRAPH_PACKAGE", "Parmetis" );
                    Zoltan_Set_Param( zoltan_lb, "OBJ_WEIGHT_DIM", "1" );
                    Zoltan_Set_Param( zoltan_lb, "EDGE_WEIGHT_DIM", "1" );
                    Zoltan_Set_Param( zoltan_lb, "CHECK_GRAPH", "0" );
                    Zoltan_Set_Param( zoltan_lb, "GRAPH_SYM_WEIGHT", "ADD" );
                    Zoltan_Set_Param( zoltan_lb, "PARMETIS_ITR", "100" );
                    break;
                case HyperGraph:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "HYPERGRAPH" );
                    Zoltan_Set_Param( zoltan_lb, "HYPERGRAPH_PACKAGE", "PHG" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_CUT_OBJECTIVE", "CONNECTIVITY" );
                    Zoltan_Set_Param( zoltan_lb, "CHECK_HYPERGRAPH", "0" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_REPART_MULTIPLIER", "100" );
                    Zoltan_Set_Param( zoltan_lb, "OBJ_WEIGHT_DIM", "1" );
                    Zoltan_Set_Param( zoltan_lb, "EDGE_WEIGHT_DIM", "0" );
                    Zoltan_Set_Param( zoltan_lb, "PHG_EDGE_WEIGHT_OPERATION", "MAX" );
                    break;
                case Hierarchical:
                    Zoltan_Set_Param( zoltan_lb, "LB_METHOD", "HIER" );
                    Zoltan_Set_Param( zoltan_lb, "HIER_DEBUG_LEVEL", "0" );
                    Zoltan_Set_Param( zoltan_lb, "DEBUG_LEVEL", "0" );
                    Zoltan_Set_Param( zoltan_lb, "OBJ_WEIGHT_DIM", "1" );
                    Zoltan_Set_Param( zoltan_lb, "EDGE_WEIGHT_DIM", "1" );
                    GenerateHiearchicalParts( );
                    Zoltan_Set_Hier_Num_Levels_Fn( zoltan_lb, &zoltan_hier_num_levels, ( void * ) this );
                    Zoltan_Set_Hier_Method_Fn( zoltan_lb, &zoltan_hier_method, ( void * ) this );
                    Zoltan_Set_Hier_Part_Fn( zoltan_lb, &zoltan_hier_part, ( void * ) this );
                    break;
            }
        }

        void FiniteStateSubset::GenerateHiearchicalParts( ) {
            //
            // Initialize hierarchical data
            //
            // Split processors based on shared memory
            MPI_Comm my_node;
            PetscMPIInt my_global_rank;
            MPI_Comm_rank( comm, &my_global_rank );
            MPI_Info mpi_info;
            MPI_Info_create( &mpi_info );
            MPI_Comm_split_type( comm, MPI_COMM_TYPE_SHARED, 0, mpi_info, &my_node );

            int my_node_rank, my_intranode_rank;
            MPI_Comm node_leader;
            int is_leader;

            // My rank within the node
            MPI_Comm_rank( my_node, &my_intranode_rank );

            // Am I the leader of my node?
            is_leader = my_intranode_rank == 0 ? 1 : 0;

            // Distinguish node leaders and others
            MPI_Comm_split( comm, is_leader, 0, &node_leader );

            // If I am the leader, my rank in the leader group is my node rank
            MPI_Comm_rank( node_leader, &my_node_rank );

            // Broadcast my node rank to the others in my node group
            MPI_Bcast( &my_node_rank, 1, MPI_INT, 0, my_node );

            my_part[ 0 ] = my_node_rank;
            my_part[ 1 ] = my_intranode_rank;

//            PetscSynchronizedPrintf(comm, "Processor %d has level 0 part %d and level 1 part %d \n", my_global_rank, my_node_rank, my_intranode_rank);
//            PetscSynchronizedFlush(comm, PETSC_STDOUT);

            MPI_Comm_free( &node_leader );
            MPI_Info_free( &mpi_info );
        }

        void FiniteStateSubset::SetHiearchicalMethods( int level, struct Zoltan_Struct *zz ) {
            switch ( level ) {
                case 0: // graph partitioning for inter-node level
                    Zoltan_Set_Param( zz, "LB_METHOD", "GRAPH" );
                    Zoltan_Set_Param( zz, "GRAPH_PACKAGE", "PARMETIS");
                    Zoltan_Set_Param( zz, "DEBUG_LEVEL", "0" );
                    Zoltan_Set_Param( zz, "CHECK_GRAPH", "0" );
                    Zoltan_Set_Param( zz, "PARMETIS_ITR", "100" );
                    Zoltan_Set_Param( zz, "GRAPH_BUILD_TYPE", "FAST_NO_DUP" );
                    break;
                case 1: // Block for intra-node level
                    Zoltan_Set_Param( zz, "LB_METHOD", "Block" );
                    Zoltan_Set_Param( zz, "DEBUG_LEVEL", "0" );
                    break;
                default:
                    break;
            }
        }

        arma::Row< PetscInt > FiniteStateSubset::State2Petsc( arma::Mat< PetscInt > &state, bool count_sinks ) {
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

            arma::Row< ZOLTAN_ID_TYPE > lids( i_vaild_constr.n_elem );
            arma::Row< int > parts( i_vaild_constr.n_elem );
            arma::Row< int > owners( i_vaild_constr.n_elem );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE >>::from(
                    state.cols( i_vaild_constr ));

            Zoltan_DD_Find( state_directory, gids.memptr( ), lids.memptr( ), nullptr, parts.memptr( ),
                            i_vaild_constr.n_elem,
                            owners.memptr( ));

            const PetscInt *starts;
            if ( count_sinks ) {
                PetscLayoutGetRanges( state_layout, &starts );
            } else {
                PetscLayoutGetRanges( state_layout_no_sinks, &starts );
            }

            for ( int i{0}; i < i_vaild_constr.n_elem; i++ ) {
                int indx = i_vaild_constr( i );
                if ( owners[ i ] >= 0 && parts[ i ] >= 0 ) {
                    indices( indx ) = starts[ parts[ i ]] + ( PetscInt ) lids( i );
                } else {
                    indices( indx ) = -1;
                }
            }

            return indices;
        }

        void FiniteStateSubset::State2Petsc( arma::Mat< PetscInt > &state, PetscInt *indx, bool count_sinks ) {

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

            arma::Row< ZOLTAN_ID_TYPE > lids( i_vaild_constr.n_elem );
            arma::Row< int > parts( i_vaild_constr.n_elem );
            arma::Row< int > owners( i_vaild_constr.n_elem );
            arma::Mat< ZOLTAN_ID_TYPE > gids = arma::conv_to< arma::Mat< ZOLTAN_ID_TYPE >>::from(
                    state.cols( i_vaild_constr ));

            Zoltan_DD_Find( state_directory, gids.memptr( ), lids.memptr( ), nullptr, parts.memptr( ),
                            i_vaild_constr.n_elem,
                            owners.memptr( ));

            const PetscInt *starts;
            if ( count_sinks ) {
                PetscLayoutGetRanges( state_layout, &starts );
            } else {
                PetscLayoutGetRanges( state_layout_no_sinks, &starts );
            }
            for ( int i{0}; i < i_vaild_constr.n_elem; i++ ) {
                auto ii = i_vaild_constr( i );
                if ( owners[ i ] >= 0 && parts[ i ] >= 0 ) {
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
                global_ids[ i ] = ( ZOLTAN_ID_TYPE ) adj_data.states_gid[ i ];
                local_ids[ i ] = ( ZOLTAN_ID_TYPE ) i;
            }
            if ( wgt_dim == 1 ) {
                for ( int i{0}; i < nstate_local; ++i ) {
                    obj_wgts[ i ] = adj_data.states_weights[ i ];
                }
            }
        }

        int FiniteStateSubset::GiveZoltanObjSize( int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                                  ZOLTAN_ID_PTR local_id, int *ierr ) {
            return 1;
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
            arma::Row< int > parts( nstate_local );
            parts.fill( my_rank );
            zoltan_err = Zoltan_DD_Update( state_directory, &local_dd_gids[ 0 ], &lids[ 0 ], nullptr, parts.memptr( ),
                                           nstate_local );
            ZOLTANCHKERRABORT( comm, zoltan_err );
            PetscLogEventEnd( total_update_dd, 0, 0, 0, 0 );
        }

        void FiniteStateSubset::UpdateStateStatus( arma::Mat< PetscInt > states, arma::Row< char > status ) {
            PetscLogEventBegin( total_update_dd, 0, 0, 0, 0 );
            int zoltan_err;

            // Update hash table
            PetscInt n_update = status.n_elem;
            if ( n_update != states.n_cols ) {
                PetscPrintf( comm, "FSS: UpdateStateStatus: states and status arrays have incompatible dimensions.\n" );
            }
            arma::Mat< ZOLTAN_ID_TYPE > local_dd_gids( n_species, n_update );

            if ( n_update > 0 ) {
                for ( PetscInt i = 0; i < n_update; ++i ) {
                    local_dd_gids.col( i ) = arma::conv_to< arma::Col< ZOLTAN_ID_TYPE>>::from(
                            states.col( i ));
                }
            }

            zoltan_err = Zoltan_DD_Update( state_directory, local_dd_gids.memptr( ), nullptr, status.memptr( ), nullptr,
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
            if ( n_update != states.n_cols ) {
                PetscPrintf( comm,
                             "FSS: UpdateStateLIDStatus: states and status arrays have incompatible dimensions.\n" );
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

            zoltan_err = Zoltan_DD_Update( state_directory, local_dd_gids.memptr( ), &lids[ 0 ], &status[ 0 ], nullptr,
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
