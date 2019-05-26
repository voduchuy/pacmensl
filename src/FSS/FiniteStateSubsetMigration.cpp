//
// Created by Huy Vo on 5/7/19.
//

#include "FiniteStateSubsetMigration.h"

namespace cme {
    namespace parallel {
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
    }
}