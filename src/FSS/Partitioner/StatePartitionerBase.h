//
// Created by Huy Vo on 5/27/19.
//

#ifndef PFSPAT_STATESETPARTITIONERBASE_H
#define PFSPAT_STATESETPARTITIONERBASE_H

#include <zoltan.h>
#include <armadillo>
#include <mpi.h>
#include "cme_util.h"
#include "string.h"

namespace cme {
    namespace parallel {
        enum PartitioningType {
            Block, Graph, HyperGraph, Hierarchical, None
        };

        enum PartitioningApproach {
            FromScratch, Repartition, Refine
        };

        class StatePartitionerBase {
        protected:
            MPI_Comm comm_;
            int my_rank_;
            int comm_size_;
            Zoltan_Struct *zoltan_lb_ = nullptr;
            arma::Mat< int > *state_ptr_ = nullptr;
            arma::Mat< int > *stoich_mat_ptr_ = nullptr;
            Zoltan_DD_Struct *state_dir_ptr_ = nullptr;
            int *layout_ = nullptr;
            int *ind_starts = nullptr;

            PartitioningType type = None;
            PartitioningApproach approach;

            int num_species_ = 0;
            int num_local_states_ = 0; ///< Number of local states held by the processor in the current partitioning

            int *states_indices_; ///< one-dimensional indices of the states
            float *states_weights_; ///< Computational weights associated with each state, here we assign to these weights the number of FLOPs needed

            virtual void set_zoltan_parameters( );

            virtual void generate_data( );

            virtual void free_data( );

            void state2ordering( arma::Mat< PetscInt > &state, PetscInt *indx );

            /* Zoltan interface functions */
            static int zoltan_num_obj( void *data, int *ierr );

            static void zoltan_obj_list( void *data, int num_gid_entries, int num_lid_entries,
                                         ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                         int *ierr );

            static int zoltan_obj_size( void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                        ZOLTAN_ID_PTR local_id, int *ierr );

            static void zoltan_pack_states( void *data, int num_gid_entries, int num_lid_entries, int num_ids,
                                            ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *dest,
                                            int *sizes, int *idx, char *buf, int *ierr );

            static void zoltan_mid_migrate_pp( void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                               ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                               int *import_procs, int *import_to_part,
                                               int num_export, ZOLTAN_ID_PTR export_global_ids,
                                               ZOLTAN_ID_PTR export_local_ids,
                                               int *export_procs, int *export_to_part, int *ierr );

            static void zoltan_unpack_states( void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                                              int *sizes, int *idx, char *buf, int *ierr );

        public:
            explicit StatePartitionerBase( MPI_Comm _comm );

            void set_lb_approach( PartitioningApproach _approach ) { approach = _approach; };

            void partition( arma::Mat< int > &states, Zoltan_DD_Struct *state_directory, arma::Mat< int > &stoich_mat,
                            int *layout );

            ~StatePartitionerBase( );
        };

        std::string part2str( PartitioningType part );

        PartitioningType str2part( std::string str );

        std::string partapproach2str( PartitioningApproach part_approach );

        PartitioningApproach str2partapproach( std::string str );
    }
}

#endif //PFSPAT_STATESETPARTITIONERBASE_H
