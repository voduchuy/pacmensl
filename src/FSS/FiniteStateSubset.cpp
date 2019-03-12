//
// Created by Huy Vo on 12/4/18.
//
#include <FSS/FiniteStateSubset.h>
#include "FiniteStateSubset.h"


namespace cme {
    namespace parallel {
        FiniteStateSubset::FiniteStateSubset(MPI_Comm new_comm, PetscInt num_species) {
            PetscErrorCode ierr;
            MPI_Comm_dup(new_comm, &comm);
            n_species = num_species;
            local_states.resize(n_species, 0);
            local_states_status.resize(0);
            nstate_global = 0;
            stoichiometry.resize(n_species, 0);

            /// Set up Zoltan load-balancing objects
            zoltan_lb = Zoltan_Create(comm);
            zoltan_explore = Zoltan_Create(comm);

            /// Set up Zoltan's parallel directory
            ierr = Zoltan_DD_Create(&state_directory, comm, n_species, 1, 0, hash_table_length, 0);

            /// Register event logging
            ierr = PetscLogEventRegister("Generate graph data", 0, &generate_graph_data);
            CHKERRABORT(comm, ierr);
            ierr = PetscLogEventRegister("Zoltan partitioning", 0, &call_partitioner);
            CHKERRABORT(comm, ierr);

            /// Set up the default FSP hyper-rectangular shape
            rhs_constr.resize(n_species);
            auto default_lhs = [this](PetscInt constr_idx, PetscInt num_states, PetscInt num_species, PetscInt *states,
                                      double *vals) {
                for (auto i{0}; i < num_states; i++) {
                    vals[i] = (double) states[i * num_species + constr_idx];
                }
            };
            lhs_constr = default_lhs;


            /// Layout of the mapping between states and Petsc ordering
            ierr = PetscLayoutCreate(comm, &state_layout);
            CHKERRABORT(comm, ierr);
        };

        void FiniteStateSubset::SetStoichiometry(arma::Mat<PetscInt> SM) {
            stoichiometry = SM;
            n_species = SM.n_rows;
            n_reactions = SM.n_cols;
            stoich_set = 1;
            local_reachable_states.set_size(n_species * n_reactions, 0);
            local_reachable_states_status.set_size(n_reactions, 0);
        }

        void FiniteStateSubset::SetLBType(PartitioningType lb_type) {
            partitioning_type = lb_type;
        }

        void FiniteStateSubset::SetRepartApproach(PartitioningApproach approach) {
            repart_approach = approach;
            switch (approach) {
                case FromScratch:
                    zoltan_part_opt = "PARTITION";
                    break;
                case Repartition:
                    zoltan_part_opt = "REPARTITION";
                    break;
                default:
                    break;
            }
        }

        void FiniteStateSubset::SetShape(
                std::function<void(PetscInt, PetscInt, PetscInt, PetscInt *, double *)> &lhs_fun,
                arma::Row<double> &rhs_bounds) {
            lhs_constr = lhs_fun;
            rhs_constr = rhs_bounds;
        }


        void FiniteStateSubset::SetShapeBounds(arma::Row<PetscInt> &rhs_bounds) {
            rhs_constr = arma::conv_to<arma::Row<double>>::from(rhs_bounds);
        }

        void FiniteStateSubset::SetShapeBounds(arma::Row<double> &rhs_bounds) {
            rhs_constr = arma::conv_to<arma::Row<double>>::from(rhs_bounds);
        }

        void FiniteStateSubset::SetInitialStates(arma::Mat<PetscInt> X0) {
            PetscErrorCode ierr;
            PetscInt my_rank;
            MPI_Comm_rank(comm, &my_rank);

            if (X0.n_rows != n_species) {
                throw std::runtime_error(
                        "SetInitialStates: number of rows in input array is not the same as the number of species.\n");
            }

            PetscPrintf(comm, "Adding initial states...\n");
            AddStates(X0);
            PetscPrintf(comm, "Initial states set...\n");
        }

        arma::Mat<PetscInt> FiniteStateSubset::GetLocalStates() {
            arma::Mat<PetscInt> states_return(n_species, nstate_local);
            for (auto j{0}; j < nstate_local; ++j) {
                for (auto i{0}; i < n_species; ++i) {
                    states_return(i, j) = (PetscInt) local_states(i, j);
                }
            }
            return states_return;
        }

        FiniteStateSubset::~FiniteStateSubset() {
            PetscMPIInt ierr;
            ierr = MPI_Comm_free(&comm);
            Zoltan_Destroy(&zoltan_lb);
            Zoltan_DD_Destroy(&state_directory);
            PetscLayoutDestroy(&state_layout);
            CHKERRABORT(comm, ierr);
            Destroy();
        }

        arma::Row<PetscReal> FiniteStateSubset::SinkStatesReduce(Vec P) {
            PetscInt ierr;

            arma::Row<PetscReal> local_sinks(rhs_constr.n_elem), global_sinks(rhs_constr.n_elem);

            PetscInt p_local_size;
            ierr = VecGetLocalSize(P, &p_local_size);
            CHKERRABORT(comm, ierr);

            if (p_local_size != local_states.n_cols + rhs_constr.n_elem) {
                printf("FiniteStateSubset::SinkStatesReduce: The layout of p and FiniteStateSubset do not match.\n");
                MPI_Abort(comm, 1);
            }

            PetscReal *p_data;
            VecGetArray(P, &p_data);
            for (auto i{0}; i < rhs_constr.n_elem; ++i) {
                local_sinks(i) = p_data[p_local_size - 1 - i];
                ierr = MPI_Allreduce(&local_sinks[i], &global_sinks[i], 1, MPIU_REAL, MPI_SUM, comm);
                CHKERRABORT(comm, ierr);
            }

            return global_sinks;
        }

        arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species) {
            MPI_Comm comm;
            PetscObjectGetComm((PetscObject) P, &comm);

            PetscReal *local_data;
            VecGetArray(P, &local_data);

            // Find the maximum size of each dimension
            arma::Row<int> fsp_dim_local(fsp.n_species), max_num_molecules(fsp.n_species);
            for (int i{0}; i < fsp.n_species; ++i) {
                fsp_dim_local(i) = arma::max(fsp.local_states.row(i));
            }
            MPI_Allreduce(&fsp_dim_local[0], &max_num_molecules[0], fsp.n_species, MPI_INT, MPI_MAX, comm);

            arma::Col<PetscReal> p_local(local_data, fsp.nstate_local, false, true);
            arma::Col<PetscReal> v(max_num_molecules(species) + 1);
            v.fill(0.0);

            for (PetscInt i{0}; i < fsp.nstate_local; ++i) {
                v(fsp.local_states(species, i)) += p_local(i);
            }

            MPI_Barrier(comm);

            arma::Col<PetscReal> w(max_num_molecules(species) + 1);
            w.fill(0.0);
            MPI_Allreduce(&v[0], &w[0], v.size(), MPI_DOUBLE, MPI_SUM, comm);

            VecRestoreArray(P, &local_data);
            return w;
        }

        void FiniteStateSubset::Destroy() {
            if (state_layout) PetscLayoutDestroy(&state_layout);
        }

        PetscInt FiniteStateSubset::GetNumGlobalStates() {
            return nstate_global;
        }

        void FiniteStateSubset::GenerateStatesAndOrdering() {
            InitZoltanParameters();
            bool frontier_empty;
            int my_rank;
            MPI_Comm_rank(comm, &my_rank);

            // Check if the set of frontier states are empty on all processors
            {
                int n1, n2;
                n1 = (int) frontier_lids.n_elem;
                MPI_Allreduce(&n1, &n2, 1, MPI_INT, MPI_MAX, comm);
                frontier_empty = (n2 == 0);
            }

            MPI_Barrier(comm);
            while (!frontier_empty) {
                // Distribute frontier states to all processors
                DistributeFrontier();

                local_states_status.elem(frontier_lids).fill(2);
                arma::uvec active_lids = frontier_lids;

                for (int ir{0}; ir < n_reactions; ++ir) {
                    arma::Mat<PetscInt> Y(n_species, active_lids.n_elem);
                    int n_add{0};
                    for (int i{0}; i < active_lids.n_elem; i++) {
                        int idx = (int) active_lids(i);
                        arma::Col<PetscInt> x(n_species);
                        for (int ii{0}; ii < n_species; ii++) {
                            x(ii) = local_reachable_states(ir * n_species + ii, idx);
                        }
                        local_reachable_states_status(ir, idx) = CheckConstraints(x);
                        if (local_reachable_states_status(ir, idx) == 0) {
                            Y.col(n_add) = x;
                            n_add += 1;
                        } else {
                            local_states_status(idx) = -1;
                        }
                    }
                    Y.resize(n_species, n_add);
                    AddStates(Y);
                }

                // Deactivate states whose neighbors have all been explored and added to the state set
                local_states_status.elem(arma::find(local_states_status == 2)).fill(0);
                frontier_lids = arma::find(local_states_status == 1);

                // Check if the set of frontier states are empty on all processors
                {
                    int n1, n2;
                    n1 = (int) frontier_lids.n_elem;
                    MPI_Allreduce(&n1, &n2, 1, MPI_INT, MPI_MAX, comm);
                    frontier_empty = (n2 == 0);
                }
            }
            // Repartition the state set
            LoadBalance();
            for (int i{0}; i < nstate_local; ++i) {
                if (local_states_status(i) == -1) {
                    local_states_status(i) = 1;
                }
            }
            frontier_lids = arma::find(local_states_status == 1);
            int nfrontier_global, nfrontier_local;
            nfrontier_local = (int) frontier_lids.n_elem;
            MPI_Allreduce(&nfrontier_local, &nfrontier_global, 1, MPI_INT, MPI_SUM, comm);
        }

        void FiniteStateSubset::DistributeFrontier() {
            // Variables to store Zoltan's output
            int zoltan_err, ierr;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            zoltan_err = Zoltan_LB_Partition(zoltan_explore, &changes, &num_gid_entries, &num_lid_entries, &num_import,
                                             &import_global_ids, &import_local_ids,
                                             &import_procs, &import_to_part, &num_export, &export_global_ids,
                                             &export_local_ids, &export_procs, &export_to_part);
            ZOLTANCHKERRABORT(comm, zoltan_err);

            nstate_local = local_states.n_cols;
            PetscMPIInt nslocal = nstate_local;
            PetscMPIInt nsglobal;
            MPI_Allreduce(&nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm);
            nstate_global = nsglobal;
            frontier_lids = arma::find(local_states_status == 1);

            ierr = PetscLayoutDestroy(&state_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutCreate(comm, &state_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(state_layout, nstate_local + rhs_constr.n_elem);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(state_layout);
            CHKERRABORT(comm, ierr);

            // Update hash table
            auto local_gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE>>::from(local_states);
            arma::Row<ZOLTAN_ID_TYPE> lids;
            if (nstate_local > 0){
                lids = arma::regspace<arma::Row<ZOLTAN_ID_TYPE>>(0, nstate_local - 1);
            } else {
                lids.set_size(0);
            }
            zoltan_err = Zoltan_DD_Update(state_directory, &local_gids[0], &lids[0], nullptr, nullptr, nstate_local);
            ZOLTANCHKERRABORT(comm, zoltan_err);

            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
        }

        void FiniteStateSubset::LoadBalance() {
            GenerateGraphData();

            // Variables to store Zoltan's output
            int zoltan_err, ierr;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            zoltan_err = Zoltan_LB_Partition(zoltan_lb, &changes, &num_gid_entries, &num_lid_entries, &num_import,
                                             &import_global_ids, &import_local_ids,
                                             &import_procs, &import_to_part, &num_export, &export_global_ids,
                                             &export_local_ids, &export_procs, &export_to_part);
            ZOLTANCHKERRABORT(comm, zoltan_err);

            nstate_local = local_states.n_cols;
            PetscMPIInt nslocal = nstate_local;
            PetscMPIInt nsglobal;
            MPI_Allreduce(&nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm);
            nstate_global = nsglobal;
            frontier_lids = arma::find(local_states_status == 1);

            ierr = PetscLayoutDestroy(&state_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutCreate(comm, &state_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(state_layout, nstate_local + rhs_constr.n_elem);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(state_layout);
            CHKERRABORT(comm, ierr);

            // Update hash table
            auto local_gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE>>::from(local_states);
            arma::Row<ZOLTAN_ID_TYPE> lids;
            if (nstate_local > 0){
                lids = arma::regspace<arma::Row<ZOLTAN_ID_TYPE>>(0, nstate_local - 1);
            } else {
                lids.set_size(0);
            }
            zoltan_err = Zoltan_DD_Update(state_directory, &local_gids[0], &lids[0], nullptr, nullptr, nstate_local);
            ZOLTANCHKERRABORT(comm, zoltan_err);

            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
        }

        void FiniteStateSubset::AddStates(arma::Mat<PetscInt> &X) {
            int zoltan_err;
            PetscErrorCode petsc_err;
            int my_rank;

            MPI_Comm_rank(comm, &my_rank);

            arma::Mat<ZOLTAN_ID_TYPE> local_gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE>>::from(X);
            arma::Row<int> owner(X.n_cols);
            arma::uvec iselect;

            // Probe if states in X are already owned by some processor
            zoltan_err = Zoltan_DD_Find(state_directory, &local_gids[0], nullptr, nullptr, nullptr, X.n_cols,
                                        &owner[0]);
            iselect = arma::find(owner == -1);
            // Shed any states that are already included
            X = X.cols(iselect);
            MPI_Barrier(comm);

            // Add local states to Zoltan directory (only 1 state from 1 processor will be added in case of overlapping)
            local_gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE>>::from(X);
            owner.resize(X.n_cols);
            zoltan_err = Zoltan_DD_Update(state_directory, &local_gids[0], nullptr, nullptr, nullptr, X.n_cols);
            ZOLTANCHKERRABORT(comm, zoltan_err);
            MPI_Barrier(comm);

            // Remove overlaps between processors
            zoltan_err = Zoltan_DD_Find(state_directory, &local_gids[0], nullptr, nullptr, nullptr, X.n_cols,
                                        &owner[0]);
            ZOLTANCHKERRABORT(comm, zoltan_err);
            MPI_Barrier(comm);
            iselect = arma::find(owner == my_rank);
            X = X.cols(iselect);

            if (X.n_cols > 0) {
                // Append X to local states
                arma::Row<PetscInt> X_status(X.n_cols);
                X_status.fill(1);
                arma::Mat<PetscInt> X_reachable(n_species * n_reactions, X.n_cols);
                for (int ir{0}; ir < n_reactions; ++ir) {
                    X_reachable.rows(arma::span(ir * n_species, (ir + 1) * n_species - 1)) =
                            X + repmat(stoichiometry.col(ir), 1, X.n_cols);
                }
                arma::Mat<PetscInt> X_reachable_status(n_reactions, X.n_cols);
                X_reachable_status.fill(0);

                local_states = arma::join_horiz(local_states, X);
                local_states_status = arma::join_horiz(local_states_status, X_status);
                local_reachable_states = arma::join_horiz(local_reachable_states, X_reachable);
                local_reachable_states_status = arma::join_horiz(local_reachable_states_status, X_reachable_status);
            }

            nstate_local = (PetscInt) local_states.n_cols;
            PetscMPIInt nslocal = nstate_local;
            PetscMPIInt nsglobal;
            MPI_Allreduce(&nslocal, &nsglobal, 1, MPI_INT, MPI_SUM, comm);
            nstate_global = nsglobal;
            // Update local ids
            local_gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE>>::from(local_states);

            arma::Row<ZOLTAN_ID_TYPE> lids;
            if (nstate_local > 0){
            lids = arma::regspace<arma::Row<ZOLTAN_ID_TYPE>>(0, nstate_local - 1);
            } else {
                lids.set_size(0);
            }
            zoltan_err = Zoltan_DD_Update(state_directory, &local_gids[0], &lids[0], nullptr, nullptr, nstate_local);
            ZOLTANCHKERRABORT(comm, zoltan_err);

            // Store the local indices of frontier states
            frontier_lids = arma::find(local_states_status == 1);

            // Update layout
            petsc_err = PetscLayoutDestroy(&state_layout);
            CHKERRABORT(comm, petsc_err);
            petsc_err = PetscLayoutCreate(comm, &state_layout);
            CHKERRABORT(comm, petsc_err);
            petsc_err = PetscLayoutSetLocalSize(state_layout, nstate_local + rhs_constr.n_elem);
            CHKERRABORT(comm, petsc_err);
            petsc_err = PetscLayoutSetUp(state_layout);
            CHKERRABORT(comm, petsc_err);
        }

        int FiniteStateSubset::CheckConstraints(arma::Col<PetscInt> &x) {
            for (int i1{0}; i1 < n_species; ++i1) {
                if (x(i1) < 0) {
                    return -1;
                }
            }
            for (int i{0}; i < rhs_constr.n_elem; ++i) {
                double fval;
                lhs_constr(i, 1, n_species, &x[0], &fval);
                if (fval > rhs_constr(i)) {
                    return i + 1;
                }
            }
            return 0;
        }

        void FiniteStateSubset::GenerateGraphData() {
            local_observable_states.set_size(n_species * n_reactions, nstate_local);
            local_observable_states_status.set_size(n_reactions, nstate_local);
            num_local_edges.set_size(nstate_local);
            num_local_edges.fill(0);
            state_weights.set_size(nstate_local);
            state_weights.fill(4.0f * n_reactions);
            for (int ir{0}; ir < n_reactions; ++ir) {
                local_observable_states.rows(arma::span(ir * n_species, (ir + 1) * n_species - 1)) =
                        local_states - repmat(stoichiometry.col(ir), 1, nstate_local);
                for (int i{0}; i < nstate_local; ++i) {
                    arma::Col<PetscInt> x(local_observable_states.colptr(i) + ir * n_species, n_species, false, false);
                    local_observable_states_status(ir, i) = CheckConstraints(x);
                    if (local_observable_states_status(ir, i) == 0) {
                        num_local_edges(i) += 1;
                        state_weights(i) += 2.0f;
                    }
                }
            }
        }

        void FiniteStateSubset::InitZoltanParameters() {
            // Parameters for state exploration load-balancing
            Zoltan_Set_Param(zoltan_explore, "NUM_GID_ENTRIES", std::to_string(n_species).c_str());
            Zoltan_Set_Param(zoltan_explore, "NUM_LID_ENTRIES", "1");
            Zoltan_Set_Param(zoltan_explore, "IMBALANCE_TOL", "1.01");
            Zoltan_Set_Param(zoltan_explore, "AUTO_MIGRATE", "1");
            Zoltan_Set_Param(zoltan_explore, "RETURN_LISTS", "ALL");
            Zoltan_Set_Param(zoltan_explore, "DEBUG_LEVEL", "0");
            Zoltan_Set_Param(zoltan_explore, "LB_METHOD", "Block");
            Zoltan_Set_Num_Obj_Fn(zoltan_explore, &zoltan_num_frontier, (void *) this);
            Zoltan_Set_Obj_List_Fn(zoltan_explore, &zoltan_frontier_list, (void *) this);
            Zoltan_Set_Obj_Size_Fn(zoltan_explore, &zoltan_obj_size, (void *) this);
            Zoltan_Set_Pack_Obj_Multi_Fn(zoltan_explore, &zoltan_pack_states, (void *) this);
            Zoltan_Set_Unpack_Obj_Multi_Fn(zoltan_explore, &zoltan_unpack_states, (void *) this);

            // Parameters for computational load-balancing
            Zoltan_Set_Param(zoltan_lb, "NUM_GID_ENTRIES", std::to_string(n_species).c_str());
            Zoltan_Set_Param(zoltan_lb, "NUM_LID_ENTRIES", "1");
            Zoltan_Set_Param(zoltan_lb, "AUTO_MIGRATE", "1");
            Zoltan_Set_Param(zoltan_lb, "RETURN_LISTS", "ALL");
            Zoltan_Set_Param(zoltan_lb, "DEBUG_LEVEL", "0");
            // This imbalance tolerance is universal for all methods
            Zoltan_Set_Param(zoltan_lb, "IMBALANCE_TOL", "1.01");
            // Register query functions to zoltan_lb
            Zoltan_Set_Num_Obj_Fn(zoltan_lb, &zoltan_num_obj, (void *) this);
            Zoltan_Set_Obj_List_Fn(zoltan_lb, &zoltan_obj_list, (void *) this);
            Zoltan_Set_Obj_Size_Fn(zoltan_lb, &zoltan_obj_size, (void *) this);
            Zoltan_Set_Pack_Obj_Multi_Fn(zoltan_lb, &zoltan_pack_states, (void *) this);
            Zoltan_Set_Unpack_Obj_Multi_Fn(zoltan_lb, &zoltan_unpack_states, (void *) this);
            Zoltan_Set_Num_Edges_Fn(zoltan_lb, &zoltan_num_edges, (void *) this);
            Zoltan_Set_Edge_List_Multi_Fn(zoltan_lb, &zoltan_get_graph_edges, (void *) this);
            Zoltan_Set_HG_Size_CS_Fn(zoltan_lb, &zoltan_get_hypergraph_size, (void *) this);
            Zoltan_Set_HG_CS_Fn(zoltan_lb, &zoltan_get_hypergraph, (void *) this);

            switch (partitioning_type) {
                case Graph:
                    Zoltan_Set_Param(zoltan_lb, "LB_METHOD", "GRAPH");
                    Zoltan_Set_Param(zoltan_lb, "GRAPH_PACKAGE", "Parmetis");
                    Zoltan_Set_Param(zoltan_lb, "PARMETIS_METHOD", "PartKway");
                    Zoltan_Set_Param(zoltan_lb, "DEBUG_LEVEL", "0");
                    Zoltan_Set_Param(zoltan_lb, "OBJ_WEIGHT_DIM", "1");
                    Zoltan_Set_Param(zoltan_lb, "EDGE_WEIGHT_DIM", "0");
                    Zoltan_Set_Param(zoltan_lb, "CHECK_GRAPH", "0");
                    Zoltan_Set_Param(zoltan_lb, "GRAPH_SYMMETRIZE", "1");
                    Zoltan_Set_Param(zoltan_lb, "PARMETIS_ITR", "1000");
                    break;
                case HyperGraph:
                    Zoltan_Set_Param(zoltan_lb, "LB_METHOD", "HYPERGRAPH");
                    Zoltan_Set_Param(zoltan_lb, "HYPERGRAPH_PACKAGE", "PHG");
                    Zoltan_Set_Param(zoltan_lb, "CHECK_HYPERGRAPH", "0");
                    Zoltan_Set_Param(zoltan_lb, "DEBUG_LEVEL", "0");
                    Zoltan_Set_Param(zoltan_lb, "PHG_REPART_MULTIPLIER", "1000");
                    Zoltan_Set_Param(zoltan_lb, "PHG_RANDOMIZE_INPUT", "1");
                    Zoltan_Set_Param(zoltan_lb, "OBJ_WEIGHT_DIM", "1");
                    Zoltan_Set_Param(zoltan_lb, "EDGE_WEIGHT_DIM", "0");// use Zoltan default hyperedge weights
                    break;
            }
        }

        arma::Row<PetscInt> FiniteStateSubset::State2Petsc(arma::Mat<PetscInt> state) {
            arma::Row<PetscInt> indices(state.n_cols);
            arma::Row<ZOLTAN_ID_TYPE> lids(state.n_cols);
            arma::Row<int> owners(state.n_cols);
            arma::Mat<ZOLTAN_ID_TYPE> gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE >>::from(state);
            Zoltan_DD_Find(state_directory, gids.memptr(), lids.memptr(), nullptr, nullptr, state.n_cols,
                           owners.memptr());
            const PetscInt *starts;
            PetscLayoutGetRanges(state_layout, &starts);
            for (int i{0}; i < state.n_cols; i++) {
                if (owners[i] != -1) {
                    indices(i) = starts[owners[i]] + (PetscInt) lids(i);
                } else {
                    indices(i) = -1;
                }
                for (int ii{0}; ii < n_species; ++ii){
                    if (state(ii, i) < 0){
                        indices(i) = -1;
                        break;
                    }
                }
            }
            return indices;
        }

        void FiniteStateSubset::State2Petsc(arma::Mat<PetscInt> state, PetscInt *indx) {
            arma::Row<ZOLTAN_ID_TYPE> lids(state.n_cols);
            arma::Row<int> owners(state.n_cols);
            arma::Mat<ZOLTAN_ID_TYPE> gids = arma::conv_to<arma::Mat<ZOLTAN_ID_TYPE >>::from(state);
            Zoltan_DD_Find(state_directory, gids.memptr(), lids.memptr(), nullptr, nullptr, state.n_cols,
                           owners.memptr());
            const PetscInt *starts;
            PetscLayoutGetRanges(state_layout, &starts);
            for (int i{0}; i < state.n_cols; i++) {
                if (owners[i] != -1) {
                    indx[i] = starts[owners[i]] + (PetscInt) lids(i);
                } else {
                    indx[i] = -1;
                }
                for (int ii{0}; ii < n_species; ++ii){
                    if (state(ii, i) < 0){
                        indx[i] = -1;
                        break;
                    }
                }
            }
        }

        std::tuple<PetscInt, PetscInt> FiniteStateSubset::GetLayoutStartEnd() {
            PetscInt start, end, ierr;
            ierr = PetscLayoutGetRange(state_layout, &start, &end);
            CHKERRABORT(comm, ierr);
            return std::make_tuple(start, end);
        }


        MPI_Comm FiniteStateSubset::GetComm() {
            return comm;
        }

        PetscInt FiniteStateSubset::GetNumLocalStates() {
            return nstate_local;
        }

        PetscInt FiniteStateSubset::GetNumSpecies() {
            return (PetscInt(n_species));
        }

        PetscInt FiniteStateSubset::GetNumReactions() {
            return stoichiometry.n_cols;
        }

        arma::Row<double> FiniteStateSubset::GetShapeBounds() {
            return arma::Row<double>(rhs_constr);
        }

        arma::Mat<PetscInt> FiniteStateSubset::GetReachableStateStatus() {
            return local_reachable_states_status;
        }

        PetscInt FiniteStateSubset::GetNumConstraints() {
            return rhs_constr.n_elem;
        }

        std::string part2str(PartitioningType part) {
            switch (part) {
                case Graph:
                    return std::string("graph");
                case HyperGraph:
                    return std::string("hyper_graph");
                default:
                    return std::string("graph");
            }
        }

        PartitioningType str2part(std::string str) {
            if (str == "graph" || str == "Graph" || str == "GRAPH") {
                return Graph;
            } else {
                return HyperGraph;
            }
        }

        std::string partapproach2str(PartitioningApproach part_approach) {
            switch (part_approach) {
                case FromScratch:
                    return std::string("from_scratch");
                case Repartition:
                    return std::string("repart");
                default:
                    return std::string("from_scratch");
            }
        }

        PartitioningApproach str2partapproach(std::string str) {
            if (str == "from_scratch" || str == "partition" || str == "FromScratch" || str == "FROMSCRATCH") {
                return FromScratch;
            } else if (str == "repart" || str == "repartition" || str == "REPARTITION" || str == "Repart" ||
                       str == "Repartition") {
                return Repartition;
            } else {
                return FromScratch;
            }
        }
    }
}
