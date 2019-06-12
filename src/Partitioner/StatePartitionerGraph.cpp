//
// Created by Huy Vo on 5/7/19.
//

#include "StatePartitionerGraph.h"

    namespace pecmeal {

        StatePartitionerGraph::StatePartitionerGraph( MPI_Comm _comm ) : StatePartitionerBase( _comm ) {}

        void StatePartitionerGraph::set_zoltan_parameters( ) {
            StatePartitionerBase::set_zoltan_parameters( );
            Zoltan_Set_Num_Edges_Fn( zoltan_lb_, &StatePartitionerGraph::zoltan_num_edges, ( void * ) this );
            Zoltan_Set_Edge_List_Multi_Fn( zoltan_lb_, &StatePartitionerGraph::zoltan_edge_list, ( void * ) this );
//            Zoltan_Set_Param( zoltan_lb_, "GRAPH_BUILD_TYPE", "FAST_NO_DUP" );
            Zoltan_Set_Param( zoltan_lb_, "LB_METHOD", "GRAPH" );
            Zoltan_Set_Param( zoltan_lb_, "GRAPH_PACKAGE", "Parmetis" );
            Zoltan_Set_Param( zoltan_lb_, "OBJ_WEIGHT_DIM", "1" );
            Zoltan_Set_Param( zoltan_lb_, "EDGE_WEIGHT_DIM", "1" );
            Zoltan_Set_Param( zoltan_lb_, "CHECK_GRAPH", "2" );
            Zoltan_Set_Param( zoltan_lb_, "GRAPH_SYM_WEIGHT", "ADD" );
            Zoltan_Set_Param( zoltan_lb_, "PARMETIS_ITR", "100" );
        }

        void StatePartitionerGraph::generate_data( ) {
            auto num_reactions = ( int ) stoich_mat_ptr_->n_cols;

            arma::Mat< int > RX( num_species_, num_local_states_ ); // states connected to local_states_tmp
            arma::Row< int > irx( num_local_states_ );

            states_indices_ = new int[num_local_states_];
            reachable_states = new int[2 * num_local_states_ * ( 1 + num_reactions )];
            states_weights_ = new float[num_local_states_];
            num_edges = new int[num_local_states_];
            edge_ptr = new int[num_local_states_];
            edge_weights = new float[2 * num_local_states_ * ( 1 + num_reactions )];
            num_reachable_states = 0;

            // find indices of the local states
            state2ordering( *state_ptr_, &states_indices_[ 0 ] );

            // Initialize graph data
            for ( auto i = 0; i < num_local_states_; ++i ) {
                edge_ptr[ i ] = i * 2 * ( 1 + num_reactions );
                num_edges[ i ] = 0;
                states_weights_[ i ] =
                        ( float ) 2.0f * num_reactions + 1.0f *
                                                         num_reactions; // Each state's weight will be added with the number of edges connected to the state in the loop below
            }
            // Enter edges and weights
            PetscInt e_ptr, edge_loc;
            // Edges corresponding to the rows of the CME
            for ( int reaction = 0; reaction < num_reactions; ++reaction ) {
                RX = *state_ptr_ - arma::repmat(( *stoich_mat_ptr_ ).col( reaction ), 1, num_local_states_ );
                state2ordering( RX, &irx[ 0 ] );

                for ( int istate = 0; istate < num_local_states_; ++istate ) {
                    if ( irx( istate ) >= 0 ) {
                        e_ptr = edge_ptr[ istate ];
                        // Edges on the row count toward the vertex weight
                        states_weights_[ istate ] +=
                                1.0f;
                        // Is the edge (istate, irx(istate)) already entered?
                        edge_loc = -1;
                        for ( int j = 0; j < num_edges[ istate ]; ++j ) {
                            if ( irx( istate ) == reachable_states[ e_ptr + j ] ) {
                                edge_loc = j;
//                                std::cout << "row: edge ( " << states_indices_[istate] << ", " << irx[istate] << ") already exists.\n";
                                break;
                            }
                        }
                        // If the edge is new, enter it to the data structure
                        if ( edge_loc == -1 ) {
                            num_edges[ istate ] += 1;
                            num_reachable_states++;
                            reachable_states[ e_ptr + num_edges[ istate ] - 1 ] = irx( istate );
                            edge_weights[ e_ptr + num_edges[ istate ] -
                                          1 ] = 1.0f;
                        }
                    }
                }
            }

            int *num_edges_row = new int[num_local_states_];
            for ( int i{0}; i < num_local_states_; ++i ) {
                num_edges_row[ i ] = num_edges[ i ];
            }
            // Edges corresponding to the columns of the CME
            for ( int reaction = 0; reaction < num_reactions; ++reaction ) {
                RX = *state_ptr_ + arma::repmat(( *stoich_mat_ptr_ ).col( reaction ), 1, num_local_states_ );
                state2ordering( RX, &irx[ 0 ] );

                for ( auto istate = 0; istate < num_local_states_; ++istate ) {
                    if ( irx[ istate ] >= 0 ) {
                        e_ptr = edge_ptr[ istate ];
                        // Is the edge (istate, irx(istate)) already entered?
                        edge_loc = -1;
                        for ( auto j = 0; j < num_edges_row[ istate ]; ++j ) {
                            if ( irx( istate ) == reachable_states[ e_ptr + j ] ) {
                                edge_loc = j;
//                                std::cout << "col: edge ( " << states_indices_[istate] << ", " << irx[istate] << ") already exists.\n";
                                break;
                            }
                        }
                        // If the edge already exists, add value
                        if ( edge_loc >= 0 ) {
                            edge_weights[ e_ptr + edge_loc ] = 2.0f;
                        } else {
                            edge_loc = -1;
                            for ( auto j = num_edges_row[ istate ]; j < num_edges[ istate ]; ++j ) {
                                if ( irx( istate ) == reachable_states[ e_ptr + j ] ) {
                                    edge_loc = j;
                                    break;
                                }
                            }
                            if ( edge_loc == -1 ) {
                                num_edges[ istate ] += 1;
                                reachable_states[ e_ptr + num_edges[ istate ] -
                                                  1 ] = irx( istate );
                                edge_weights[ e_ptr + num_edges[ istate ] - 1 ] = 1.0f;
                                num_reachable_states++;
                            }
                        }
                    }
                }
            }
            delete[] num_edges_row;
        }

        void StatePartitionerGraph::free_data( ) {
            StatePartitionerBase::free_data( );
            delete[] num_edges;
            delete[] reachable_states;
            delete[] edge_weights;
            delete[] edge_ptr;
        }

        int
        StatePartitionerGraph::zoltan_num_edges( void *data, int num_gid_entries, int num_lid_entries,
                                                 ZOLTAN_ID_PTR global_id,
                                                 ZOLTAN_ID_PTR local_id, int *ierr ) {
            *ierr = ZOLTAN_OK;
            return (( StatePartitionerGraph * ) data )->num_edges[ *local_id ];
        }

        void StatePartitionerGraph::zoltan_edge_list( void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                                      ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
                                                      int *num_edges,
                                                      ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                                                      float *ewgts, int *ierr ) {

            auto my_data = ( StatePartitionerGraph * ) data;
            if ( my_data->num_local_states_ != num_obj ) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            if (( num_gid_entries != 1 ) || ( num_lid_entries != 1 )) {
                *ierr = ZOLTAN_FATAL;
                return;
            }

            int k = 0;
            for ( int iobj = 0; iobj < num_obj; ++iobj ) {
                int ptr = my_data->edge_ptr[ iobj ];
                for ( auto i = 0; i < num_edges[ iobj ]; ++i ) {
                    nbor_global_id[ k ] = ( ZOLTAN_ID_TYPE ) my_data->reachable_states[ ptr + i ];
                    if ( wgt_dim == 1 ) {
                        ewgts[ k ] = my_data->edge_weights[ ptr + i ];
                    }
                    k++;
                }
            }
            *ierr = ZOLTAN_OK;
        }
    }