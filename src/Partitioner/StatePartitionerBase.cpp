//
// Created by Huy Vo on 5/27/19.
//

#include "StatePartitionerBase.h"

namespace pacmensl {
    StatePartitionerBase::StatePartitionerBase(MPI_Comm _comm) {
        comm_ = _comm;
        zoltan_lb_ = Zoltan_Create(comm_);
        MPI_Comm_rank(comm_, &my_rank_);
        MPI_Comm_size(comm_, &comm_size_);
        ind_starts = new int[comm_size_];
    }

    int StatePartitionerBase::partition(arma::Mat<int> &states, Zoltan_DD_Struct *state_directory,
                                        arma::Mat<int> &stoich_mat, int *layout) {
        state_ptr_ = &states;
        state_dir_ptr_ = state_directory;
        stoich_mat_ptr_ = &stoich_mat;
        num_species_ = (int) states.n_rows;
        num_local_states_ = (int) states.n_cols;
        layout_ = layout;

        ind_starts[0] = 0;
        for (int i{1}; i < comm_size_; ++i) {
            ind_starts[i] = ind_starts[i - 1] + layout[i - 1];
        }

        set_zoltan_parameters();
        generate_data();
        // Variables to store Zoltan's output
        int zoltan_err;
        int changes, num_gid_entries, num_lid_entries, num_import, num_export;
        ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
        int *import_procs, *import_to_part, *export_procs, *export_to_part;

        zoltan_err = Zoltan_LB_Partition(zoltan_lb_, &changes, &num_gid_entries, &num_lid_entries, &num_import,
                                         &import_global_ids, &import_local_ids,
                                         &import_procs, &import_to_part, &num_export, &export_global_ids,
                                         &export_local_ids, &export_procs, &export_to_part);
        ZOLTANCHKERRQ(zoltan_err);
        Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
        Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
        free_data();
        return 0;
    }

    void StatePartitionerBase::generate_data() {
        states_indices_ = new int[num_local_states_];
        states_weights_ = new float[num_local_states_];
        auto num_reactions = ( int ) stoich_mat_ptr_->n_cols;

        arma::Mat< int > RX( num_species_, num_local_states_ ); // states_ connected to local_states_tmp
        arma::Row< int > irx( num_local_states_ );

        // find indices of the local states_
        state2ordering( *state_ptr_, &states_indices_[ 0 ] );

        // Initialize graph data
        for ( auto i = 0; i < num_local_states_; ++i ) {
            states_weights_[ i ] = 1.0f; // Each state's weight will be added with the number of edges connected to the state in the loop below
        }
    };

    void StatePartitionerBase::free_data() {
        delete[] states_indices_;
        delete[] states_weights_;
    }

    StatePartitionerBase::~StatePartitionerBase() {
        delete[] ind_starts;
        Zoltan_Destroy(&zoltan_lb_);
        comm_ = MPI_COMM_NULL;
    }

    void StatePartitionerBase::state2ordering(arma::Mat<PetscInt> &state, PetscInt *indx) {

        int num_species = state.n_rows;

        arma::Row<PetscInt> ipositive(state.n_cols);
        ipositive.fill(0);

        for (int i{0}; i < ipositive.n_cols; ++i) {
            for (int j = 0; j < num_species; ++j) {
                if (state(j, i) < 0) {
                    ipositive(i) = -1;
                    indx[i] = -1;
                    break;
                }
            }
        }

        arma::uvec i_vaild_constr = arma::find(ipositive == 0);

        arma::Row<ZOLTAN_ID_TYPE> state_indices(i_vaild_constr.n_elem);
        arma::Row<int> parts(i_vaild_constr.n_elem);
        arma::Row<int> owners(i_vaild_constr.n_elem);
        arma::Mat<ZOLTAN_ID_TYPE> gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE >>::from(
                state.cols(i_vaild_constr));

        Zoltan_DD_Find(state_dir_ptr_, gids.memptr(), state_indices.memptr(), nullptr, parts.memptr(),
                       (int) i_vaild_constr.n_elem,
                       owners.memptr());

        for (int i{0}; i < i_vaild_constr.n_elem; i++) {
            auto ii = i_vaild_constr(i);
            if (owners[i] >= 0 && parts[i] >= 0) {
                indx[ii] = ind_starts[parts[i]] + state_indices(i);
            } else {
                indx[ii] = -1;
            }
        }
    }

    void StatePartitionerBase::set_zoltan_parameters() {
        // Parameters for computational load-balancing
        Zoltan_Set_Param(zoltan_lb_, "NUM_GID_ENTRIES", "1");
        Zoltan_Set_Param(zoltan_lb_, "NUM_LID_ENTRIES", "1");
        Zoltan_Set_Param(zoltan_lb_, "AUTO_MIGRATE", "1");
        Zoltan_Set_Param(zoltan_lb_, "RETURN_LISTS", "ALL");
        Zoltan_Set_Param(zoltan_lb_, "DEBUG_LEVEL", "0");
        Zoltan_Set_Param(zoltan_lb_, "IMBALANCE_TOL", "1.01");
        Zoltan_Set_Param(zoltan_lb_, "LB_METHOD", "BLOCK");
        Zoltan_Set_Param(zoltan_lb_, "LB_APPROACH", partapproach2str(approach).c_str());
        Zoltan_Set_Param(zoltan_lb_, "SCATTER_GRAPH", "2");
        // Register query functions to zoltan_lb_
        Zoltan_Set_Num_Obj_Fn(zoltan_lb_, &StatePartitionerBase::zoltan_num_obj, (void *) this);
        Zoltan_Set_Obj_List_Fn(zoltan_lb_, &StatePartitionerBase::zoltan_obj_list, (void *) this);
        Zoltan_Set_Obj_Size_Fn(zoltan_lb_, &StatePartitionerBase::zoltan_obj_size, (void *) this);
        Zoltan_Set_Pack_Obj_Multi_Fn(zoltan_lb_, &StatePartitionerBase::zoltan_pack_states, (void *) this);
        Zoltan_Set_Unpack_Obj_Multi_Fn(zoltan_lb_, &StatePartitionerBase::zoltan_unpack_states, (void *) this);
        Zoltan_Set_Mid_Migrate_PP_Fn(zoltan_lb_, &StatePartitionerBase::zoltan_mid_migrate_pp, (void *) this);
    }

    int StatePartitionerBase::zoltan_num_obj(void *data, int *ierr) {
        return ((StatePartitionerBase *) data)->num_local_states_;
    }

    void StatePartitionerBase::zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries,
                                               ZOLTAN_ID_PTR global_ids,
                                               ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                                               int *ierr) {

        auto *my_data = (StatePartitionerBase *) data;
        for (int i{0}; i < my_data->num_local_states_; ++i) {
            global_ids[i] = (ZOLTAN_ID_TYPE) my_data->states_indices_[i];
            local_ids[i] = (ZOLTAN_ID_TYPE) i;
        }
        if (wgt_dim == 1) {
            for (int i{0}; i < my_data->num_local_states_; ++i) {
                obj_wgts[i] = my_data->states_weights_[i];
            }
        }
    }

    int StatePartitionerBase::zoltan_obj_size(void *data, int num_gid_entries, int num_lid_entries,
                                              ZOLTAN_ID_PTR global_id,
                                              ZOLTAN_ID_PTR local_id, int *ierr) {
        auto my_data = (StatePartitionerBase *) data;
        *ierr = ZOLTAN_OK;
        return my_data->num_species_ * sizeof(int);
    }

    void
    StatePartitionerBase::zoltan_pack_states(void *data, int num_gid_entries, int num_lid_entries, int num_ids,
                                             ZOLTAN_ID_PTR global_ids,
                                             ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx,
                                             char *buf,
                                             int *ierr) {
        auto my_data = (StatePartitionerBase *) data;
        for (int i{0}; i < num_ids; ++i) {
            auto ptr = (int *) &buf[idx[i]];
            auto state_id = local_ids[i];
            for (int j{0}; j < my_data->num_species_; ++j) {
                *(ptr + j) = (*my_data->state_ptr_)(j, state_id);
            }
        }
        *ierr = ZOLTAN_OK;
    }

    void StatePartitionerBase::zoltan_mid_migrate_pp(void *data, int num_gid_entries, int num_lid_entries,
                                                     int num_import,
                                                     ZOLTAN_ID_PTR import_global_ids,
                                                     ZOLTAN_ID_PTR import_local_ids,
                                                     int *import_procs, int *import_to_part, int num_export,
                                                     ZOLTAN_ID_PTR export_global_ids,
                                                     ZOLTAN_ID_PTR export_local_ids,
                                                     int *export_procs, int *export_to_part, int *ierr) {
        auto my_data = (StatePartitionerBase *) data;
        // remove the packed states_ from local data structure
        arma::uvec i_keep(my_data->num_local_states_);
        i_keep.zeros();
        for (int i{0}; i < num_export; ++i) {
            i_keep(export_local_ids[i]) = 1;
        }
        i_keep = arma::find(i_keep == 0);

        *my_data->state_ptr_ = my_data->state_ptr_->cols(i_keep);
        my_data->num_local_states_ = (int) my_data->state_ptr_->n_cols;

        // allocate more space for states_ to be imported
        my_data->state_ptr_->resize(my_data->num_species_, my_data->num_local_states_ + num_import);
    }

    void
    StatePartitionerBase::zoltan_unpack_states(void *data, int num_gid_entries, int num_ids,
                                               ZOLTAN_ID_PTR global_ids,
                                               int *sizes, int *idx, char *buf, int *ierr) {
        auto my_data = (StatePartitionerBase *) data;
        // Unpack new local states_
        for (int i{0}; i < num_ids; ++i) {
            auto ptr = (int *) &buf[idx[i]];
            for (int j{0}; j < my_data->num_species_; ++j) {
                (*my_data->state_ptr_)(j, my_data->num_local_states_ + i) = *(ptr + j);
            }
        }
        my_data->num_local_states_ += num_ids;
    }

    std::string part2str(PartitioningType part) {
        switch (part) {
            case PartitioningType::GRAPH:
                return std::string("Graph");
            case PartitioningType::HYPERGRAPH:
                return std::string("Hypergraph");
          case PartitioningType::HIERARCHICAL:
                return std::string("Hiearchical");
            default:
                return std::string("Block");
        }
    }

    PartitioningType str2part(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        if (str == "graph") {
            return PartitioningType::GRAPH;
        } else if (str == "hypergraph") {
            return PartitioningType::HYPERGRAPH;
        } else if (str == "hier" || str == "hierarchical") {
            return PartitioningType::HIERARCHICAL;
        } else {
            return PartitioningType::BLOCK;
        }
    }

    std::string partapproach2str(PartitioningApproach part_approach) {
        switch (part_approach) {
            case PartitioningApproach::FROMSCRATCH:
                return std::string("partition");
            case PartitioningApproach::REPARTITION:
                return std::string("repartition");
            default:
                return std::string("refine");
        }
    }

    PartitioningApproach str2partapproach(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        if (str == "from_scratch" || str == "partition") {
            return PartitioningApproach::FROMSCRATCH;
        } else if (str == "repart" || str == "repartition") {
            return PartitioningApproach::REPARTITION;
        } else {
            return PartitioningApproach::REFINE;
        }
    }
}