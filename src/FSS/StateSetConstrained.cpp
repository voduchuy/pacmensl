//
// Created by Huy Vo on 5/31/19.
//

#include "StateSetBase.h"
#include "StateSetConstrained.h"

cme::parallel::StateSetConstrained::StateSetConstrained( MPI_Comm new_comm, int num_species,
                                                               cme::parallel::PartitioningType lb_type,
                                                               cme::parallel::PartitioningApproach lb_approach )
        : StateSetBase( new_comm, num_species,
                           lb_type, lb_approach ) {

    StateSetConstrained::rhs_constr.resize( num_species );
    StateSetConstrained::lhs_constr = StateSetConstrained::default_constr_fun;
}

/**
 * Call level: local.
 * @param x
 * @return 0 if x satisfies all constraints; otherwise -1.
 */
PetscInt cme::parallel::StateSetConstrained::check_state( PetscInt *x ) {
    for ( int i1{0}; i1 < num_species_; ++i1 ) {
        if ( x[ i1 ] < 0 ) {
            return -1;
        }
    }
    int *fval;
    fval = new int[rhs_constr.n_elem];
    lhs_constr( num_species_, rhs_constr.n_elem, 1, x, &fval[ 0 ] );

    for ( int i{0}; i < rhs_constr.n_elem; ++i ) {
        if ( fval[ i ] > rhs_constr( i )) {
            return -1;
        }
    }
    delete[] fval;
    return 0;
}

/**
 * Call level: local.
 * @param num_states : number of states. x : array of states. satisfied: output array of size num_states*num_constraints.
 * @return void.
 */
void
cme::parallel::StateSetConstrained::check_constraint_on_proc( PetscInt num_states, PetscInt *x,
                                                                 PetscInt *satisfied ) {
    auto *fval = new int[num_states * rhs_constr.n_elem];
    lhs_constr( num_species_, rhs_constr.n_elem, num_states, x, fval );
    for ( int iconstr = 0; iconstr < rhs_constr.n_elem; ++iconstr ) {
        for ( int i = 0; i < num_states; ++i ) {
            satisfied[ num_states * iconstr + i ] = ( fval[ rhs_constr.n_elem * i + iconstr ] <=
                                                      rhs_constr( iconstr )) ? 1 : 0;
            for ( int j = 0; j < num_species_; ++j ) {
                if ( x[ num_species_ * i + j ] < 0 ) {
                    satisfied[ num_states * iconstr + i ] = 1;
                }
            }
        }
    }
    delete[] fval;
}

arma::Row< int > cme::parallel::StateSetConstrained::get_shape_bounds( ) {
    return arma::Row< int >( rhs_constr );
}

PetscInt cme::parallel::StateSetConstrained::get_num_constraints( ) {
    return rhs_constr.n_elem;
}

void
cme::parallel::StateSetConstrained::default_constr_fun( int num_species, int num_constr, int n_states, int *states,
                                                           int *outputs ) {
    for ( int i{0}; i < n_states * num_species; ++i ) {
        outputs[ i ] = states[ i ];
    }
}

void cme::parallel::StateSetConstrained::set_shape(
        fsp_constr_multi_fn *lhs_fun,
        arma::Row< int > &rhs_bounds ) {
    lhs_constr = lhs_fun;
    rhs_constr = rhs_bounds;
}

void cme::parallel::StateSetConstrained::set_shape_bounds( arma::Row< int > &rhs_bounds ) {
    rhs_constr = rhs_bounds;
}

/**
 * Call level: collective.
 * This function also distribute the states into the processors to improve the load-balance of matrix-vector multplications.
 */
void cme::parallel::StateSetConstrained::expand( ) {
    bool frontier_empty;

    // Switch states with status -1 to 1, for they may expand to new states when the shape constraints are relaxed
    retrieve_state_status( );
    arma::uvec iupdate = find( local_states_status_ == -1 );
    arma::Mat< int > states_update;
    arma::Row< char > new_status;
    if ( iupdate.n_elem > 0 ) {
        new_status.set_size( iupdate.n_elem );
        new_status.fill( 1 );
        states_update = local_states_.cols( iupdate );
    } else {
        states_update.set_size( num_species_, 0 );
        new_status.set_size( 0 );
    }
    update_state_status( states_update, new_status );

    retrieve_state_status( );
    frontier_lids_ = find( local_states_status_ == 1 );

    logger_.event_begin( logger_.state_exploration_event );
    // Check if the set of frontier states are empty on all processors
    {
        int n1, n2;
        n1 = ( int ) frontier_lids_.n_elem;
        MPI_Allreduce( &n1, &n2, 1, MPI_INT, MPI_MAX, comm_ );
        frontier_empty = ( n2 == 0 );
    }

    arma::Row< char > frontier_status;
    while ( !frontier_empty ) {
        // Distribute frontier states to all processors
        frontiers_ = local_states_.cols( frontier_lids_ );
        distribute_frontiers( );
        frontier_status.set_size( frontiers_.n_cols );
        frontier_status.fill( 0 );

        arma::Mat< PetscInt > Y( num_species_, frontiers_.n_cols * num_reactions_ );
        arma::Row< PetscInt > ystatus( Y.n_cols );

        for ( int i{0}; i < frontiers_.n_cols; i++ ) {
            for ( int j{0}; j < num_reactions_; ++j ) {
                Y.col( j * frontiers_.n_cols + i ) = frontiers_.col( i ) + stoichiometry_matrix_.col( j );
                ystatus( j * frontiers_.n_cols + i ) = check_state( Y.colptr( j * frontiers_.n_cols + i ));

                if ( ystatus( j * frontiers_.n_cols + i ) < 0 ) {
                    frontier_status( i ) = -1;
                }
            }
        }
        Y = Y.cols( find( ystatus == 0 ));
        Y = unique_columns( Y );
        add_states( Y );

        // Deactivate states whose neighbors have all been explored and added to the state set
        update_state_status( frontiers_, frontier_status );
        retrieve_state_status( );
        frontier_lids_ = find( local_states_status_ == 1 );

        // Check if the set of frontier states are empty on all processors
        {
            int n1, n2;
            n1 = ( int ) frontier_lids_.n_elem;
            MPI_Allreduce( &n1, &n2, 1, MPI_INT, MPI_MAX, comm_ );
            frontier_empty = ( n2 == 0 );
        }
    }
    logger_.event_end( logger_.state_exploration_event );

    logger_.event_begin( logger_.call_partitioner_event );
    if ( comm_size_ > 1 ) {
        if ( num_global_states_old_ * ( 1.0 + lb_threshold_ ) <= 1.0 * num_global_states_ || num_global_states_old_ == 0 ) {
            num_global_states_old_ = num_global_states_;
            PetscPrintf( comm_, "Repartitioning...\n" );
            load_balance( );
        }
    }
    logger_.event_end( logger_.call_partitioner_event );
}
