//
// Created by Huy Vo on 5/7/19.
//

#include "StatePartitionerHyperGraph.h"

namespace cme{
    namespace parallel{

        void StatePartitionerHyperGraph::generate_data( ) {
            auto num_reactions = ( int ) stoich_mat_ptr_->n_cols;

            arma::Mat< int > RX(( size_t ) num_species_,
                                     ( size_t ) num_local_states_ ); // states connected to local_states_tmp
            arma::Row< int > irx( num_local_states_ );
            states_indices_ = new int[num_local_states_];

            states_weights_ = new float[num_local_states_];
            num_edges = new int[num_local_states_];
            edge_ptr = new int[num_local_states_+1];
            reachable_states = new int[2 * num_local_states_ * ( 1 + stoich_mat_ptr_->n_cols )];
            edge_weights = new float[2 * num_local_states_ * ( 1 + stoich_mat_ptr_->n_cols )];
            num_reachable_states = 0;

            // Find global indices of local states
            state2ordering( *state_ptr_, &states_indices_[ 0 ] );

            // Initialize hypergraph data
            for ( auto i = 0; i < num_local_states_; ++i ) {
                edge_ptr[ i ] = i * 2 * ( 1 + num_reactions );
                num_edges[ i ] = 0;
                edge_weights[ i ] = 0.0f;
                states_weights_[ i ] =
                        ( float ) 2.0f * num_reactions + 1.0f * num_reactions;
            }
            // Enter edges and weights
            PetscInt e_ptr, edge_loc;
            arma::uvec i_neg;
            // Edges corresponding to the rows of the CME
            for ( auto reaction = 0; reaction < num_reactions; ++reaction ) {
                RX = *state_ptr_ - arma::repmat(( *stoich_mat_ptr_ ).col( reaction ), 1, num_local_states_ );
                state2ordering( RX, &irx[ 0 ] );

                for ( auto i = 0; i < num_local_states_; ++i ) {
                    if ( irx[i] >= 0 ) {
                        e_ptr = edge_ptr[ i ];
                        states_weights_[ i ] +=
                                1.0f;
                        // Is the edge (i, irx(i)) already entered?
                        edge_loc = -1;
                        for ( auto j = 0; j < num_edges[ i ]; ++j ) {
                            if ( irx( i ) == reachable_states[ e_ptr + j ] ) {
                                edge_loc = j;
                                break;
                            }
                        }
                        // If the edge already exists, do nothing
                        if ( edge_loc < 0 ){
                            num_edges[ i ] += 1;
                            num_reachable_states++;
                            reachable_states[ e_ptr + num_edges[ i ] - 1 ] = irx( i );
                        }
                    }
                }
            }
            edge_ptr[ num_local_states_ ] = num_reachable_states + num_local_states_;
        }

        void StatePartitionerHyperGraph::set_zoltan_parameters( ) {
            StatePartitionerBase::set_zoltan_parameters( );
            Zoltan_Set_HG_Size_CS_Fn( zoltan_lb_, &StatePartitionerHyperGraph::zoltan_hg_size, ( void * ) this );
            Zoltan_Set_HG_CS_Fn( zoltan_lb_, &StatePartitionerHyperGraph::zoltan_hg_edge_list, ( void * ) this );
            Zoltan_Set_HG_Size_Edge_Wts_Fn( zoltan_lb_, &StatePartitionerHyperGraph::zoltan_hg_size_eweights, ( void * ) this );
            Zoltan_Set_HG_Edge_Wts_Fn( zoltan_lb_, &StatePartitionerHyperGraph::zoltan_hg_edge_weights, ( void * ) this );
            Zoltan_Set_Param( zoltan_lb_, "LB_METHOD", "HYPERGRAPH" );
            Zoltan_Set_Param( zoltan_lb_, "HYPERGRAPH_PACKAGE", "PHG" );
            Zoltan_Set_Param( zoltan_lb_, "PHG_CUT_OBJECTIVE", "CONNECTIVITY" );
            Zoltan_Set_Param( zoltan_lb_, "CHECK_HYPERGRAPH", "0" );
            Zoltan_Set_Param( zoltan_lb_, "PHG_REPART_MULTIPLIER", "100" );
            Zoltan_Set_Param( zoltan_lb_, "OBJ_WEIGHT_DIM", "1" );
            Zoltan_Set_Param( zoltan_lb_, "EDGE_WEIGHT_DIM", "0" );
            Zoltan_Set_Param( zoltan_lb_, "PHG_EDGE_WEIGHT_OPERATION", "MAX" );
        }

        void StatePartitionerHyperGraph::free_data( ) {
            StatePartitionerBase::free_data();
            delete[] num_edges;
            Zoltan_Free(( void ** ) &states_indices_, __FILE__, __LINE__ );
            Zoltan_Free(( void ** ) &reachable_states, __FILE__, __LINE__ );
            delete[] edge_ptr;
        }

        void StatePartitionerHyperGraph::zoltan_hg_size( void *data, int *num_lists, int *num_pins, int *format, int *ierr ) {
            auto my_data = (StatePartitionerHyperGraph*) data;
            *num_lists = my_data->num_local_states_;
            *num_pins = my_data->num_reachable_states + my_data->num_local_states_;
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void StatePartitionerHyperGraph::zoltan_hg_edge_list( void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                                      ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr,
                                                      ZOLTAN_ID_PTR pin_gid, int *ierr ) {
            auto my_data = (StatePartitionerHyperGraph*) data;
            if (( num_vertices != my_data->num_local_states_ ) || ( num_pins != my_data->num_reachable_states + num_vertices) ||
                ( format != ZOLTAN_COMPRESSED_VERTEX )) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            int k = 0;
            for ( int i{0}; i < num_vertices; ++i ) {
                vtx_gid[ i ] = ( ZOLTAN_ID_TYPE ) my_data->states_indices_[ i ];
                vtx_edge_ptr[ i ] = (i==0)? 0 : vtx_edge_ptr[i-1] + my_data->num_edges[i-1] + 1;
                pin_gid[k] = (ZOLTAN_ID_TYPE) my_data->states_indices_[i]; k++;
                for ( int j{0}; j < my_data->num_edges[i];++j){
                    pin_gid[ k ] = ( ZOLTAN_ID_TYPE ) my_data->reachable_states[ my_data->edge_ptr[i] + j ];
                    k++;
                }
            }
            *ierr = ZOLTAN_OK;
        }

        void StatePartitionerHyperGraph::zoltan_hg_size_eweights( void *data, int *num_edges, int *ierr ) {
            auto my_data = (StatePartitionerHyperGraph*) data;
            *num_edges = my_data->num_local_states_;
            *ierr = ZOLTAN_OK;
        }

        void StatePartitionerHyperGraph::zoltan_hg_edge_weights( void *data, int num_gid_entries, int num_lid_entries, int num_edges,
                                                            int edge_weight_dim, ZOLTAN_ID_PTR edge_GID,
                                                            ZOLTAN_ID_PTR edge_LID, float *edge_weight, int *ierr ) {
            auto my_data = (StatePartitionerHyperGraph*) data;
            for ( int i{0}; i < num_edges; ++i ) {
                edge_GID[ i ] = (ZOLTAN_ID_TYPE) my_data->states_indices_[ i ];
                edge_LID[ i ] = (ZOLTAN_ID_TYPE) i;
                edge_weight[ i ] = 1.0f;
            }
            *ierr = ZOLTAN_OK;
        }
    }
}