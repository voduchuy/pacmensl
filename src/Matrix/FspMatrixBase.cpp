#include <Matrix/FspMatrixBase.h>
#include <petscblaslapack_stdcall.h>

#include "Matrix/FspMatrixBase.h"
#include "Sys.h"
#include "FspMatrixBase.h"

namespace pacmensl {

    FspMatrixBase::FspMatrixBase( MPI_Comm comm ) {
        MPI_Comm_dup( comm, &comm_ );
        MPI_Comm_rank( comm_, &my_rank_ );
        MPI_Comm_size( comm_, &comm_size_ );
    }

    int FspMatrixBase::action( PetscReal t, Vec x, Vec y ) {
        Int ierr;

        ierr = t_fun_( t , num_reactions_, time_coefficients_.memptr(), t_fun_args_);
        PACMENSLCHKERRQ(ierr);

        ierr = VecSet( y, 0.0 );
        CHKERRQ(ierr);

        ierr = VecGetLocalVector( x, xx );
        CHKERRQ(ierr);
        ierr = VecGetLocalVector( y, yy );
        CHKERRQ(ierr);

        ierr = VecScatterBegin( action_ctx_, x, lvec_, INSERT_VALUES, SCATTER_FORWARD );
        CHKERRQ(ierr);
        for ( Int ir{0}; ir < num_reactions_; ++ir ) {
            ierr = MatMult( diag_mats_[ ir ], xx, zz );
            CHKERRQ(ierr);

            ierr = VecAXPY( yy, time_coefficients_[ ir ], zz );
            CHKERRQ(ierr);
        }
        ierr = VecScatterEnd( action_ctx_, x, lvec_, INSERT_VALUES, SCATTER_FORWARD );
        CHKERRQ(ierr);

        for ( Int ir{0}; ir < num_reactions_; ++ir ) {
            ierr = MatMult( offdiag_mats_[ ir ], lvec_, zz );
            CHKERRQ(ierr);

            ierr = VecAXPY( yy, time_coefficients_[ ir ], zz );
            CHKERRQ(ierr);
        }

        ierr = VecRestoreLocalVector( x, xx );
        CHKERRQ(ierr);
        ierr = VecRestoreLocalVector( y, yy );
        CHKERRQ(ierr);
    }

    int FspMatrixBase::GenerateValues( const StateSetBase &fsp, const arma::Mat< Int > &SM, const PropFun &propensity,
                                       void *propensity_args, const TcoefFun &new_t_fun, void *t_fun_args ) {
        DetermineLayout_( fsp );

        PetscInt ierr;
        PetscMPIInt rank;
        PetscInt n_species, n_local_states, own_start, own_end;
        const arma::Mat< Int > &state_list = fsp.GetStatesRef( );
        arma::Mat< Int > can_reach_my_state( state_list.n_rows, state_list.n_cols );

        // arrays for counting nonzero entries on the diagonal and off-diagonal blocks
        arma::Mat< Int > d_nnz, o_nnz;
        // global indices of off-processor entries needed for matrix-vector product
        arma::Row< Int > out_indices;
        // arrays of nonzero column indices
        arma::Mat< Int > irnz, irnz_off;
        // array o fmatrix values
        arma::Mat< PetscReal > mat_vals;

        ISLocalToGlobalMapping local2global_rows, local2global_lvec;

        n_species = fsp.GetNumSpecies( );
        n_local_states = fsp.GetNumLocalStates( );
        num_reactions_ = fsp.GetNumReactions( );
        t_fun_ = new_t_fun;
        t_fun_args_ = t_fun_args;
        diag_mats_.resize( num_reactions_ );
        offdiag_mats_.resize( num_reactions_ );
        time_coefficients_.set_size(num_reactions_);

        MPI_Comm_rank( comm_, &rank );

        ierr = VecGetOwnershipRange( work_, &own_start, &own_end );
        CHKERRQ(ierr);

        // Get the global and local numbers of rows
        ierr = VecGetSize( work_, &n_rows_global_ );
        CHKERRQ(ierr);

        // Find the nnz per row of diagonal and off-diagonal matrices
        irnz.set_size( n_local_states, num_reactions_ );
        irnz_off.set_size( n_local_states, num_reactions_ );
        out_indices.set_size( n_local_states * num_reactions_ );
        mat_vals.set_size( n_local_states, num_reactions_ );

        d_nnz.set_size( n_rows_local_, num_reactions_ );
        o_nnz.set_size( n_rows_local_, num_reactions_ );
        d_nnz.fill( 1 );
        o_nnz.zeros( );
        int out_count = 0;
        irnz_off.fill( -1 );

        // Count nnz for matrix rows
        for ( auto i_reaction{0}; i_reaction < num_reactions_; ++i_reaction ) {
            can_reach_my_state = state_list - arma::repmat( SM.col( i_reaction ), 1, state_list.n_cols );
            fsp.State2Index( can_reach_my_state, irnz.colptr( i_reaction ));
            propensity( i_reaction, can_reach_my_state.n_rows, can_reach_my_state.n_cols, &can_reach_my_state[ 0 ],
                        mat_vals.colptr( i_reaction ), propensity_args );

            for ( auto i_state{0}; i_state < n_local_states; ++i_state ) {
                if ( irnz( i_state, i_reaction ) >= own_start && irnz( i_state, i_reaction ) < own_end ) {
                    d_nnz( i_state, i_reaction ) += 1;
                } else if ( irnz( i_state, i_reaction ) >= 0 ) {
                    irnz_off( i_state, i_reaction ) = irnz( i_state, i_reaction );
                    irnz( i_state, i_reaction ) = -1;
                    o_nnz( i_state, i_reaction ) += 1;
                    out_indices( out_count ) = irnz_off( i_state, i_reaction );
                    out_count += 1;
                }
            }
        }

        // Create mapping from global rows to local rows
        PetscInt *my_global_indices = new PetscInt[n_rows_local_];
        for ( auto i{0}; i < n_rows_local_; ++i ) {
            my_global_indices[ i ] = own_start + i;
        }
        ierr = ISLocalToGlobalMappingCreate( comm_, 1, n_rows_local_, my_global_indices, PETSC_COPY_VALUES,
                                             &local2global_rows );
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingSetType( local2global_rows, ISLOCALTOGLOBALMAPPINGHASH );
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingSetFromOptions( local2global_rows );
        CHKERRQ(ierr);
        delete[] my_global_indices;

        // Create mapping from local ghost vec to global indices
        out_indices.resize( out_count );
        arma::Row< Int > out_indices2 = arma::unique( out_indices );
        out_count = 0;
        for ( auto i{0}; i < out_indices2.n_elem; ++i ) {
            if ( out_indices2[ i ] < own_start || out_indices2[ i ] >= own_end ) {
                out_indices( out_count ) = out_indices2[ i ];
                out_count += 1;
            }
        }

        lvec_length_ = PetscInt( out_count );
        ierr = VecCreateSeq(PETSC_COMM_SELF, lvec_length_, &lvec_ );
        CHKERRQ(ierr);
        ierr = VecSetUp( lvec_ );
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreate( comm_, 1, lvec_length_, &out_indices[ 0 ], PETSC_COPY_VALUES,
                                             &local2global_lvec );
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingSetType( local2global_lvec, ISLOCALTOGLOBALMAPPINGHASH );
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingSetFromOptions( local2global_lvec );
        CHKERRQ(ierr);

        // Create vecscatter for collecting off-diagonal vector entries
        IS from_is;
        ierr = ISCreateGeneral( comm_, lvec_length_, &out_indices[ 0 ], PETSC_COPY_VALUES, &from_is );
        CHKERRQ(ierr);
        ierr = VecScatterCreate( work_, from_is, lvec_, NULL, &action_ctx_ );
        CHKERRQ(ierr);
        ierr = ISDestroy( &from_is );
        CHKERRQ(ierr);

        // Generate local vectors for matrix action
        VecCreateSeq(PETSC_COMM_SELF, n_rows_local_, &xx );
        VecSetUp( xx );
        VecCreateSeq(PETSC_COMM_SELF, n_rows_local_, &yy );
        VecSetUp( yy );
        VecCreateSeq(PETSC_COMM_SELF, n_rows_local_, &zz );
        VecSetUp( zz );

        // Generate values for diagonal and off-diagonal blocks
        // Convert the global indices of nonzero entries to local indices
        ierr = ISGlobalToLocalMappingApply( local2global_rows, IS_GTOLM_MASK, n_local_states * num_reactions_,
                                            irnz.memptr( ), NULL, irnz.memptr( ));
        CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply( local2global_lvec, IS_GTOLM_MASK, n_local_states * num_reactions_,
                                            irnz_off.memptr( ), NULL, irnz_off.memptr( ));
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy( &local2global_lvec );
        CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy( &local2global_rows );
        CHKERRQ(ierr);

        arma::Col< PetscReal > diag_vals( n_local_states );
        for ( PetscInt i_reaction{0}; i_reaction < num_reactions_; ++i_reaction ) {
            ierr = MatCreate(PETSC_COMM_SELF, &diag_mats_[ i_reaction ] );
            CHKERRQ(ierr);
            ierr = MatSetType( diag_mats_[ i_reaction ], MATSEQAIJ );
            CHKERRQ(ierr);
            ierr = MatSetSizes( diag_mats_[ i_reaction ], n_rows_local_, n_rows_local_, n_rows_local_, n_rows_local_ );
            CHKERRQ(ierr);
            ierr = MatSeqAIJSetPreallocation( diag_mats_[ i_reaction ], NULL, d_nnz.colptr( i_reaction ));
            CHKERRQ(ierr);

            ierr = MatCreate(PETSC_COMM_SELF, &offdiag_mats_[ i_reaction ] );
            CHKERRQ(ierr);
            ierr = MatSetType( offdiag_mats_[ i_reaction ], MATSEQAIJ );
            CHKERRQ(ierr);
            ierr = MatSetSizes( offdiag_mats_[ i_reaction ], n_rows_local_, lvec_length_, n_rows_local_, lvec_length_ );
            CHKERRQ(ierr);
            ierr = MatSeqAIJSetPreallocation( offdiag_mats_[ i_reaction ], NULL, o_nnz.colptr( i_reaction ));
            CHKERRQ(ierr);

            propensity( i_reaction, n_species, n_local_states, &state_list[ 0 ], &diag_vals[ 0 ], propensity_args );
            for ( auto i_state{0}; i_state < n_local_states; ++i_state ) {
                // Set values for the diagonal block
                ierr = MatSetValue( diag_mats_[ i_reaction ], i_state, i_state, -1.0 * diag_vals[ i_state ],
                                    INSERT_VALUES );
                CHKERRQ(ierr);
                ierr = MatSetValue( diag_mats_[ i_reaction ], i_state, irnz( i_state, i_reaction ),
                                    mat_vals( i_state, i_reaction ), INSERT_VALUES );
                CHKERRQ(ierr);

                // Set values for the off-diagonal block
                ierr = MatSetValue( offdiag_mats_[ i_reaction ], i_state, irnz_off( i_state, i_reaction ),
                                    mat_vals( i_state, i_reaction ), INSERT_VALUES );
                CHKERRQ(ierr);
            }

            ierr = MatAssemblyBegin( diag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
            CHKERRQ(ierr);
            ierr = MatAssemblyEnd( diag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
            CHKERRQ(ierr);
            ierr = MatAssemblyBegin( offdiag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
            CHKERRQ(ierr);
            ierr = MatAssemblyEnd( offdiag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
            CHKERRQ(ierr);
        }
        return 0;
    }

    FspMatrixBase::~FspMatrixBase( ) {
        Destroy( );
        if (comm_ != nullptr) MPI_Comm_free( &comm_ );
    }

    int FspMatrixBase::Destroy( ) {
        PetscErrorCode ierr;
        for ( PetscInt i{0}; i < num_reactions_; ++i ) {
            if ( diag_mats_[ i ] != nullptr ) {
                ierr = MatDestroy( &diag_mats_[ i ] );
                CHKERRQ(ierr);
            }
            if ( offdiag_mats_[ i ] != nullptr ) {
                ierr = MatDestroy( &offdiag_mats_[ i ] );
                CHKERRQ(ierr);
            }
        }
        if ( xx != nullptr ) {
            ierr = VecDestroy( &xx );
            CHKERRQ(ierr);
        }
        if ( yy != nullptr ) {
            ierr = VecDestroy( &yy );
            CHKERRQ(ierr);
        }
        if ( zz != nullptr ) {
            ierr = VecDestroy( &zz );
            CHKERRQ(ierr);
        }
        if ( work_ != nullptr ) {
            ierr = VecDestroy( &work_ );
            CHKERRQ(ierr);
        }
        if ( lvec_ != nullptr ) {
            ierr = VecDestroy( &lvec_ );
            CHKERRQ(ierr);
        }
        if ( action_ctx_ != nullptr ) {
            ierr = VecScatterDestroy( &action_ctx_ );
            CHKERRQ(ierr);
        }

        xx = nullptr;
        yy = nullptr;
        zz = nullptr;
        work_ = nullptr;
        lvec_ = nullptr;
        action_ctx_ = nullptr;
    }

    PetscInt FspMatrixBase::get_local_ghost_length( ) const {
        return lvec_length_;
    }

    int FspMatrixBase::DetermineLayout_( const StateSetBase &fsp ) {
        PetscErrorCode ierr;
        try {
            n_rows_local_ = fsp.GetNumLocalStates( );
            ierr = 0;
        }
        catch(std::runtime_error& ex){
            ierr = -1;
        }
        PACMENSLCHKERRQ(ierr);

        // Generate matrix layout from FSP's layout
        ierr = VecCreate( comm_, &work_ );
        CHKERRQ(ierr);
        ierr = VecSetFromOptions( work_ );
        CHKERRQ(ierr);
        ierr = VecSetSizes( work_, n_rows_local_, PETSC_DECIDE );
        CHKERRQ(ierr);
        ierr = VecSetUp( work_ );
        CHKERRQ(ierr);
        return 0;
    }
}
