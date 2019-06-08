#include <Matrix/FspMatrixBase.h>
#include <petscblaslapack_stdcall.h>

#include "Matrix/FspMatrixBase.h"
#include "util/cme_util.h"
#include "FspMatrixBase.h"


namespace cme {
    namespace parallel {

        FspMatrixBase::FspMatrixBase( MPI_Comm comm ) {
            MPI_Comm_dup( comm, &comm_ );
            MPI_Comm_rank( comm_, &my_rank_ );
            MPI_Comm_size( comm_, &comm_size_ );
        }

        void FspMatrixBase::action( PetscReal t, Vec x, Vec y ) {
            Int ierr;
            ierr = VecSet( y, 0.0 );
            CHKERRABORT( comm_, ierr );

            ierr = VecGetLocalVectorRead( x, xx );
            CHKERRABORT( comm_, ierr );
            ierr = VecGetLocalVectorRead( y, yy );
            CHKERRABORT( comm_, ierr );
            arma::Row< Real > coefficients = t_fun_( t );

            ierr = VecScatterBegin( action_ctx_, x, lvec_, INSERT_VALUES, SCATTER_FORWARD );
            CHKERRABORT( comm_, ierr );
            for ( Int ir{0}; ir < n_reactions_; ++ir ) {
                ierr = MatMult( diag_mats_[ ir ], xx, zz );
                CHKERRABORT( comm_, ierr );

                ierr = VecAXPY( yy, coefficients[ ir ], zz );
                CHKERRABORT( comm_, ierr );
            }
            ierr = VecScatterEnd( action_ctx_, x, lvec_, INSERT_VALUES, SCATTER_FORWARD );
            CHKERRABORT( comm_, ierr );

            for ( Int ir{0}; ir < n_reactions_; ++ir ) {
                ierr = MatMult( offdiag_mats_[ ir ], lvec_, zz );
                CHKERRABORT( comm_, ierr );

                ierr = VecAXPY( yy, coefficients[ ir ], zz );
                CHKERRABORT( comm_, ierr );
            }

            ierr = VecRestoreLocalVectorRead( x, xx );
            CHKERRABORT( comm_, ierr );
            ierr = VecRestoreLocalVectorRead( y, yy );
            CHKERRABORT( comm_, ierr );
        }

        void FspMatrixBase::generate_values( const StateSetBase &fsp, const arma::Mat< Int > &SM, PropFun prop,
                                             TcoefFun new_t_fun ) {
            determine_layout(fsp);

            PetscInt ierr;
            PetscMPIInt rank;
            PetscInt n_local_states, own_start, own_end;
            const arma::Mat< Int > &state_list = fsp.GetStatesRef();
            arma::Mat< Int > can_reach_my_state( state_list.n_rows, state_list.n_cols );

            // arrays for counting nonzero entries on the diagonal and off-diagonal blocks
            arma::Mat< Int > d_nnz, o_nnz;
            // array of global indices of off-diagonal entries needed for matrix-vector product
            arma::Row< Int > out_indices;
            // arrays of nonzero column indices
            arma::Mat< Int > irnz, irnz_off;
            // array o fmatrix values
            arma::Mat< PetscReal > mat_vals;

            ISLocalToGlobalMapping local2global_rows, local2global_lvec;

            n_local_states = fsp.GetNumLocalStates();
            n_reactions_ = fsp.GetNumReactions();
            t_fun_ = new_t_fun;
            diag_mats_.resize( n_reactions_ );
            offdiag_mats_.resize( n_reactions_ );

            MPI_Comm_rank( comm_, &rank );

            ierr = VecGetOwnershipRange( work_, &own_start, &own_end );
            CHKERRABORT( comm_, ierr );

            // Get the global and local numbers of rows
            ierr = VecGetSize( work_, &n_rows_global_ );
            CHKERRABORT( comm_, ierr );

            // Find the nnz per row of diagonal and off-diagonal matrices
            irnz.set_size( n_local_states, n_reactions_ );
            irnz_off.set_size( n_local_states, n_reactions_ );
            out_indices.set_size( n_local_states * n_reactions_ );
            mat_vals.set_size( n_local_states, n_reactions_ );

            d_nnz.set_size( n_rows_local_, n_reactions_ );
            o_nnz.set_size( n_rows_local_, n_reactions_ );
            d_nnz.fill( 1 );
            o_nnz.zeros( );
            int out_count = 0;
            irnz_off.fill( -1 );

            // Count nnz for matrix rows
            for ( auto i_reaction{0}; i_reaction < n_reactions_; ++i_reaction ) {
                can_reach_my_state = state_list - arma::repmat( SM.col( i_reaction ), 1, state_list.n_cols );
                fsp.State2Index(can_reach_my_state, irnz.colptr(i_reaction));

                for ( auto i_state{0}; i_state < n_local_states; ++i_state ) {
                    if ( irnz( i_state, i_reaction ) >= own_start && irnz( i_state, i_reaction ) < own_end ) {
                        d_nnz( i_state, i_reaction ) += 1;
                        mat_vals( i_state, i_reaction ) = prop( can_reach_my_state.colptr( i_state ), i_reaction );
                    } else if ( irnz( i_state, i_reaction ) >= 0 ) {
                        irnz_off( i_state, i_reaction ) = irnz( i_state, i_reaction );
                        irnz( i_state, i_reaction ) = -1;
                        mat_vals( i_state, i_reaction ) = prop( can_reach_my_state.colptr( i_state ), i_reaction );
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
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingSetType( local2global_rows, ISLOCALTOGLOBALMAPPINGHASH );
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingSetFromOptions( local2global_rows );
            CHKERRABORT( comm_, ierr );
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
            CHKERRABORT( comm_, ierr );
            ierr = VecSetUp( lvec_ );
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingCreate( comm_, 1, lvec_length_, &out_indices[ 0 ], PETSC_COPY_VALUES,
                                                 &local2global_lvec );
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingSetType( local2global_lvec, ISLOCALTOGLOBALMAPPINGHASH );
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingSetFromOptions( local2global_lvec );
            CHKERRABORT( comm_, ierr );

            // Create vecscatter for collecting off-diagonal vector entries
            IS from_is;
            ierr = ISCreateGeneral( comm_, lvec_length_, &out_indices[ 0 ], PETSC_COPY_VALUES, &from_is );
            CHKERRABORT( comm_, ierr );
            ierr = VecScatterCreate( work_, from_is, lvec_, NULL, &action_ctx_ );
            CHKERRABORT( comm_, ierr );
            ierr = ISDestroy( &from_is );
            CHKERRABORT( comm_, ierr );

            // Generate local vectors for matrix action
            VecCreateSeq(PETSC_COMM_SELF, n_rows_local_, &xx );
            VecSetUp( xx );
            VecCreateSeq(PETSC_COMM_SELF, n_rows_local_, &yy );
            VecSetUp( yy );
            VecCreateSeq(PETSC_COMM_SELF, n_rows_local_, &zz );
            VecSetUp( zz );

            // Generate values for diagonal and off-diagonal blocks
            // Convert the global indices of nonzero entries to local indices
            ierr = ISGlobalToLocalMappingApply( local2global_rows, IS_GTOLM_MASK, n_local_states * n_reactions_,
                                                irnz.memptr( ), NULL, irnz.memptr( ));
            CHKERRABORT( comm_, ierr );
            ierr = ISGlobalToLocalMappingApply( local2global_lvec, IS_GTOLM_MASK, n_local_states * n_reactions_,
                                                irnz_off.memptr( ), NULL, irnz_off.memptr( ));
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingDestroy( &local2global_lvec );
            CHKERRABORT( comm_, ierr );
            ierr = ISLocalToGlobalMappingDestroy( &local2global_rows );
            CHKERRABORT( comm_, ierr );

            for ( PetscInt i_reaction{0}; i_reaction < n_reactions_; ++i_reaction ) {
                ierr = MatCreate(PETSC_COMM_SELF, &diag_mats_[ i_reaction ] );
                CHKERRABORT( comm_, ierr );
                ierr = MatSetType( diag_mats_[ i_reaction ], MATSEQAIJ );
                CHKERRABORT( comm_, ierr );
                ierr = MatSetSizes( diag_mats_[ i_reaction ], n_rows_local_, n_rows_local_, n_rows_local_,
                                    n_rows_local_ );
                CHKERRABORT( comm_, ierr );
                ierr = MatSeqAIJSetPreallocation( diag_mats_[ i_reaction ], NULL, d_nnz.colptr( i_reaction ));
                CHKERRABORT( comm_, ierr );

                ierr = MatCreate(PETSC_COMM_SELF, &offdiag_mats_[ i_reaction ] );
                CHKERRABORT( comm_, ierr );
                ierr = MatSetType( offdiag_mats_[ i_reaction ], MATSEQAIJ );
                CHKERRABORT( comm_, ierr );
                ierr = MatSetSizes( offdiag_mats_[ i_reaction ], n_rows_local_, lvec_length_, n_rows_local_,
                                    lvec_length_ );
                CHKERRABORT( comm_, ierr );
                ierr = MatSeqAIJSetPreallocation( offdiag_mats_[ i_reaction ], NULL, o_nnz.colptr( i_reaction ));
                CHKERRABORT( comm_, ierr );

                for ( auto i_state{0}; i_state < n_local_states; ++i_state ) {
                    // Set values for the diagonal block
                    PetscReal diag_val = -1.0 * prop( state_list.colptr( i_state ), i_reaction );
                    ierr = MatSetValue( diag_mats_[ i_reaction ], i_state, i_state, diag_val, INSERT_VALUES );
                    CHKERRABORT( comm_, ierr );
                    ierr = MatSetValue( diag_mats_[ i_reaction ], i_state, irnz( i_state, i_reaction ),
                                        mat_vals( i_state, i_reaction ), INSERT_VALUES );
                    CHKERRABORT( comm_, ierr );

                    // Set values for the off-diagonal block
                    ierr = MatSetValue( offdiag_mats_[ i_reaction ], i_state, irnz_off( i_state, i_reaction ),
                                        mat_vals( i_state, i_reaction ), INSERT_VALUES );
                    CHKERRABORT( comm_, ierr );
                }

                ierr = MatAssemblyBegin( diag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
                CHKERRABORT( comm_, ierr );
                ierr = MatAssemblyEnd( diag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
                CHKERRABORT( comm_, ierr );
                ierr = MatAssemblyBegin( offdiag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
                CHKERRABORT( comm_, ierr );
                ierr = MatAssemblyEnd( offdiag_mats_[ i_reaction ], MAT_FINAL_ASSEMBLY );
                CHKERRABORT( comm_, ierr );
            }
        }

        FspMatrixBase::~FspMatrixBase( ) {
            MPI_Comm_free( &comm_ );
            destroy( );
        }

        void FspMatrixBase::destroy( ) {
            for ( PetscInt i{0}; i < n_reactions_; ++i ) {
                if ( diag_mats_[ i ] != NULL ) {
                    MatDestroy( &diag_mats_[ i ] );
                }
                if ( offdiag_mats_[ i ] != NULL) {
                    MatDestroy( &offdiag_mats_[ i ] );
                }
            }
            if ( xx  != NULL ) VecDestroy( &xx );
            if ( yy != NULL ) VecDestroy( &yy );
            if ( zz  != NULL) VecDestroy( &zz );
            if ( work_ != NULL ) VecDestroy( &work_ );
            if ( lvec_  != NULL) VecDestroy( &lvec_ );
            if ( action_ctx_ != NULL ) VecScatterDestroy( &action_ctx_ );
            xx = NULL;
            yy = NULL;
            zz = NULL;
            work_ = NULL;
            lvec_ = NULL;
            action_ctx_ = NULL;
        }

        PetscInt FspMatrixBase::get_local_ghost_length( ) const {
            return lvec_length_;
        }

        void FspMatrixBase::determine_layout( const StateSetBase &fsp ) {
            PetscErrorCode ierr;

            n_rows_local_ = fsp.GetNumLocalStates();

            // Generate matrix layout from FSP's layout
            ierr = VecCreate( comm_, &work_ );
            CHKERRABORT( comm_, ierr );
            ierr = VecSetFromOptions( work_ );
            CHKERRABORT( comm_, ierr );
            ierr = VecSetSizes( work_, n_rows_local_, PETSC_DECIDE );
            CHKERRABORT( comm_, ierr );
            ierr = VecSetUp( work_ );
            CHKERRABORT( comm_, ierr );
        }
    }
}
