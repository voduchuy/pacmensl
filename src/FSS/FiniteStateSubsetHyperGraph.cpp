//
// Created by Huy Vo on 5/7/19.
//

#include "FiniteStateSubsetHyperGraph.h"

namespace cme{
    namespace parallel{

        void FiniteStateSubset::GenerateHyperGraphData( ) {
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

            adj_data.edge_ptr = new int[n_local_tmp+1];

            adj_data.reachable_states = new PetscInt[2 * n_local_tmp * ( 1 + stoichiometry.n_cols )];

            adj_data.edge_weights = new float[2 * n_local_tmp * ( 1 + stoichiometry.n_cols )];

            adj_data.num_local_states = n_local_tmp;

            adj_data.num_reachable_states = 0;

            // Lexicographic indices of the local states
            State2Petsc( local_states, &adj_data.states_gid[ 0 ], false );

            // Initialize hypergraph data
            for ( auto i = 0; i < n_local_tmp; ++i ) {
                adj_data.edge_ptr[ i ] = i * 2 * ( 1 + n_reactions );
                adj_data.num_edges[ i ] = 0;
                adj_data.edge_weights[ i ] = 0.0f;
                adj_data.states_weights[ i ] =
                        ( float ) 2.0f * n_reactions + 1.0f *
                                                       n_reactions; // Each state's weight will be added with the number of edges connected to the state in the loop below
            }
            // Enter edges and weights
            PetscInt e_ptr, edge_loc;
            arma::uvec i_neg;
            // Edges corresponding to the rows of the CME
            for ( auto reaction = 0; reaction < n_reactions; ++reaction ) {
                RX = local_states - arma::repmat( stoichiometry.col( reaction ), 1, n_local_tmp );
                State2Petsc( RX, &irx[ 0 ], false );

                for ( auto i = 0; i < n_local_tmp; ++i ) {
                    if ( irx( i ) >= 0 ) {
                        e_ptr = adj_data.edge_ptr[ i ];
                        adj_data.states_weights[ i ] +=
                                1.0f;
                        // Is the edge (i, irx(i)) already entered?
                        edge_loc = -1;
                        for ( auto j = 0; j < adj_data.num_edges[ i ]; ++j ) {
                            if ( irx( i ) == adj_data.reachable_states[ e_ptr + j ] ) {
                                edge_loc = j;
                                break;
                            }
                        }
                        // If the edge already exists, do nothing
                        if ( edge_loc < 0 ){
                            adj_data.num_edges[ i ] += 1;
                            adj_data.num_reachable_states++;
                            adj_data.reachable_states[ e_ptr + adj_data.num_edges[ i ] - 1 ] = irx( i );
                        }
                    }
                }
            }
            adj_data.edge_ptr[ n_local_tmp ] = adj_data.num_reachable_states + adj_data.num_local_states;

            ierr = PetscLogEventEnd( generate_graph_data, 0, 0, 0, 0 );
            CHKERRABORT( comm, ierr );
        }

        void FiniteStateSubset::FreeHyperGraphData( ) {
            delete[] adj_data.num_edges;
            Zoltan_Free(( void ** ) &adj_data.states_gid, __FILE__, __LINE__ );
            Zoltan_Free(( void ** ) &adj_data.reachable_states, __FILE__, __LINE__ );
            delete[] adj_data.edge_ptr;
            delete[] adj_data.states_weights;
        }

        void FiniteStateSubset::GiveZoltanHypergraphSize( int *num_lists, int *num_pins, int *format, int *ierr ) {
            *num_lists = adj_data.num_local_states;
            *num_pins = adj_data.num_reachable_states + adj_data.num_local_states;
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubset::GiveZoltanHypergraph( int num_gid_entries, int num_vertices, int num_pins, int format,
                                                      ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr,
                                                      ZOLTAN_ID_PTR pin_gid, int *ierr ) {
            if (( num_vertices != adj_data.num_local_states ) || ( num_pins != adj_data.num_reachable_states + num_vertices) ||
                ( format != ZOLTAN_COMPRESSED_VERTEX )) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            int k = 0;
            for ( int i{0}; i < num_vertices; ++i ) {
                vtx_gid[ i ] = ( ZOLTAN_ID_TYPE ) adj_data.states_gid[ i ];
                vtx_edge_ptr[ i ] = (i==0)? 0 : vtx_edge_ptr[i-1] + adj_data.num_edges[i-1] + 1;
                pin_gid[k] = adj_data.states_gid[i]; k++;
                for ( int j{0}; j < adj_data.num_edges[i];++j){
                    pin_gid[ k ] = ( ZOLTAN_ID_TYPE ) adj_data.reachable_states[ adj_data.edge_ptr[i] + j ];
                    k++;
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void
        FiniteStateSubset::GiveZoltanHypergraphEdgeWeights( int num_gid_entries, int num_lid_entries, int num_edges,
                                                            int edge_weight_dim, ZOLTAN_ID_PTR edge_GID,
                                                            ZOLTAN_ID_PTR edge_LID, float *edge_weight, int *ierr ) {
            for ( int i{0}; i < num_edges; ++i ) {
                edge_GID[ i ] = adj_data.states_gid[ i ];
                edge_LID[ i ] = i;
                edge_weight[ i ] = 1.0f;
            }
            *ierr = ZOLTAN_OK;
        }
    }
}