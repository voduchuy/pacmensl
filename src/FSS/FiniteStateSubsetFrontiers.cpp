//
// Created by Huy Vo on 5/7/19.
//

#include "FiniteStateSubsetFrontiers.h"

namespace cme{
    namespace parallel{

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
                *ptr = local_frontier_gids( local_ids[ i ] );
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
    }
}