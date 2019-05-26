//
// Created by Huy Vo on 5/7/19.
//

#include "FiniteStateSubsetGraph.h"

namespace cme{
    namespace parallel{
        void FiniteStateSubset::GenerateGraphData( ) {

            UpdateMaxNumMolecules( );
            PetscErrorCode ierr;

            ierr = PetscLogEventBegin( generate_graph_data, 0, 0, 0, 0 );
            CHKERRABORT( comm, ierr );

            auto n_local_tmp = ( PetscInt ) local_states.n_cols;
            auto n_reactions = ( PetscInt ) stoichiometry.n_cols;

            arma::Mat< PetscInt > RX(( size_t ) n_species,
                                     ( size_t ) n_local_tmp ); // states connected to local_states_tmp
            arma::Row< PetscInt > irx( n_local_tmp );

            adj_data.states_gid = new PetscInt[n_local_tmp];

            adj_data.states_weights = new float[n_local_tmp];

            adj_data.num_edges = new int[n_local_tmp];

            adj_data.edge_ptr = new int[n_local_tmp];

            adj_data.reachable_states = new PetscInt[2 * n_local_tmp * ( 1 + n_reactions )];

            adj_data.edge_weights = new float[2 * n_local_tmp * ( 1 + n_reactions )];

            adj_data.num_local_states = n_local_tmp;

            adj_data.num_reachable_states = 0;

            // Lexicographic indices of the local states
            State2Petsc( local_states, &adj_data.states_gid[ 0 ], false );

            // Initialize graph data
            for ( auto i = 0; i < n_local_tmp; ++i ) {
                adj_data.edge_ptr[ i ] = i * 2 * ( 1 + n_reactions );
                adj_data.num_edges[ i ] = 0;
                adj_data.states_weights[ i ] =
                        ( float ) 2.0f * n_reactions + 1.0f *
                                                       n_reactions; // Each state's weight will be added with the number of edges connected to the state in the loop below
            }
            // Enter edges and weights
            PetscInt e_ptr, edge_loc;
            arma::uvec i_neg;
            // Edges corresponding to the rows of the CME
            for ( int reaction = 0; reaction < n_reactions; ++reaction ) {
                RX = local_states - arma::repmat( stoichiometry.col( reaction ), 1, n_local_tmp );
                State2Petsc( RX, &irx[ 0 ], false );

                for ( auto istate = 0; istate < n_local_tmp; ++istate ) {
                    if ( irx( istate ) >= 0 ) {
                        e_ptr = adj_data.edge_ptr[ istate ];
                        // Edges on the row count toward the vertex weight
                        adj_data.states_weights[ istate ] +=
                                1.0f;
                        // Is the edge (istate, irx(istate)) already entered?
                        edge_loc = -1;
                        for ( auto j = 0; j < adj_data.num_edges[ istate ]; ++j ) {
                            if ( irx( istate ) == adj_data.reachable_states[ e_ptr + j ] ) {
                                edge_loc = j;
                                break;
                            }
                        }
                        // If the edge is new, enter it to the data structure
                        if ( edge_loc == -1 ) {
                            adj_data.num_edges[ istate ] += 1;
                            adj_data.num_reachable_states++;
                            adj_data.reachable_states[ e_ptr + adj_data.num_edges[ istate ] - 1 ] = irx( istate );
                            adj_data.edge_weights[ e_ptr + adj_data.num_edges[ istate ] -
                                                   1 ] = 1.0f;
                        }
                    }
                }
            }

            int* num_edges_row = new int[n_local_tmp];
            for (int i{0}; i < n_local_tmp; ++i){
                num_edges_row[i] = adj_data.num_edges[i];
            }
            // Edges corresponding to the columns of the CME
            for ( int reaction = 0; reaction < n_reactions; ++reaction ) {
                RX = local_states + arma::repmat( stoichiometry.col( reaction ), 1, n_local_tmp );
                State2Petsc( RX, &irx[ 0 ], false );

                for ( auto i = 0; i < n_local_tmp; ++i ) {
                    if ( irx( i ) >= 0 ) {
                        e_ptr = adj_data.edge_ptr[ i ];
                        // Is the edge (i, irx(i)) already entered?
                        edge_loc = -1;
                        for ( auto j = 0; j < num_edges_row[ i ]; ++j ) {
                            if ( irx( i ) == adj_data.reachable_states[ e_ptr + j ] ) {
                                edge_loc = j;
                                break;
                            }
                        }
                        // If the edge already exists, add value
                        if ( edge_loc >= 0) {
                            adj_data.edge_weights[ e_ptr + edge_loc ] = 2.0f;
                        } else {
                            edge_loc = -1;
                            for ( auto j = num_edges_row[i]; j < adj_data.num_edges[ i ]; ++j ) {
                                if ( irx( i ) == adj_data.reachable_states[ e_ptr + j ] ) {
                                    edge_loc = j;
                                    break;
                                }
                            }
                            if (edge_loc == -1){
                                adj_data.num_edges[ i ] += 1;
                                adj_data.reachable_states[ e_ptr + adj_data.num_edges[ i ] - 1 ] = irx( i );
                                adj_data.edge_weights[ e_ptr + adj_data.num_edges[ i ] - 1 ] = 1.0f;
                                adj_data.num_reachable_states++;
                            }
                        }
                    }
                }
            }
            delete[] num_edges_row;
            ierr = PetscLogEventEnd( generate_graph_data, 0, 0, 0, 0 );
            CHKERRABORT( comm, ierr );
        }

        void FiniteStateSubset::FreeGraphData( ) {
            delete[] adj_data.num_edges;
            Zoltan_Free(( void ** ) &adj_data.states_gid, __FILE__, __LINE__ );
            Zoltan_Free(( void ** ) &adj_data.reachable_states, __FILE__, __LINE__ );
            delete[] adj_data.edge_ptr;
            delete[] adj_data.states_weights;
            delete[] adj_data.edge_weights;
        }


        int FiniteStateSubset::GiveZoltanNumEdges( int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                                   ZOLTAN_ID_PTR local_id, int *ierr ) {
            *ierr = ZOLTAN_OK;
            return adj_data.num_edges[ *local_id ];
        }

        void FiniteStateSubset::GiveZoltanGraphEdges( int num_gid_entries, int num_lid_entries, int num_obj,
                                                      ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *num_edges,
                                                      ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                                                      float *ewgts, int *ierr ) {

            if ( nstate_local != num_obj ) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            if (( num_gid_entries != 1 ) || ( num_lid_entries != 1 )) {
                *ierr = ZOLTAN_FATAL;
                return;
            }

            int k = 0;
            for ( int iobj = 0; iobj < num_obj; ++iobj ) {
                int edge_ptr = adj_data.edge_ptr[ iobj ];
                for ( auto i = 0; i < adj_data.num_edges[ iobj ]; ++i ) {
                    nbor_global_id[ k ] = ( ZOLTAN_ID_TYPE ) adj_data.reachable_states[ edge_ptr + i ];
                    if ( wgt_dim == 1 ) {
                        ewgts[ k ] = adj_data.edge_weights[ edge_ptr + i ];
                    }
                    k++;
                }
            }
            *ierr = ZOLTAN_OK;
        }
    }
}