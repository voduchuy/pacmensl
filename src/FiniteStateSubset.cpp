//
// Created by Huy Vo on 12/4/18.
//
#include <FiniteStateSubset.h>

namespace cme {
    namespace parallel {
        FiniteStateSubset::FiniteStateSubset(MPI_Comm new_comm) {
            int ierr;
            MPI_Comm_dup(new_comm, &comm);
            zoltan = Zoltan_Create(comm);
            partitioning_type = NotSet;
            local_states.resize(0);
            fsp_size.resize(0);
            n_states_global = 0;
            stoichiometry.resize(0);

            Zoltan_Set_Param(zoltan, "IMBALANCE_TOL", "1.01");

            Zoltan_Set_Num_Obj_Fn(zoltan, &zoltan_num_obj, (void *) &this->adj_data);
            Zoltan_Set_Obj_List_Fn(zoltan, &zoltan_obj_list, (void *) &this->adj_data);
            Zoltan_Set_Num_Geom_Fn(zoltan, &zoltan_num_geom, (void *) this);
            Zoltan_Set_Geom_Multi_Fn(zoltan, &zoltan_geom_multi, (void *) this);
            Zoltan_Set_Num_Edges_Fn(zoltan, &zoltan_num_edges, (void *) &this->adj_data);
            Zoltan_Set_Edge_List_Fn(zoltan, &zoltan_edge_list, (void *) &this->adj_data);
            Zoltan_Set_HG_Size_CS_Fn(zoltan, &zoltan_get_hypergraph_size, (void *) &this->adj_data);
            Zoltan_Set_HG_CS_Fn(zoltan, &zoltan_get_hypergraph, (void *) &this->adj_data);
        };

        void FiniteStateSubset::SetStoichiometry(arma::Mat<PetscInt> SM) {
            stoichiometry = SM;
            stoich_set = 1;
        }

        void FiniteStateSubset::SetSize(arma::Row<PetscInt> new_fsp_size) {
            fsp_size = new_fsp_size;
            n_species = (PetscInt) fsp_size.n_elem;
            n_states_global = arma::prod(fsp_size + 1);
        }

        arma::Mat<PetscInt> FiniteStateSubset::GetLocalStates() {
            arma::Mat<PetscInt> states_return = local_states;
            return states_return;
        }

        arma::Row<PetscInt> FiniteStateSubset::State2Petsc(arma::Mat<PetscInt> state) {
            arma::Row<PetscInt> lex_indices = cme::sub2ind_nd(fsp_size, state);
            std::vector<PetscBool> negative(lex_indices.n_elem);
            for (auto i{0}; i < negative.size(); ++i) {
                if (lex_indices(i) < 0) {
                    negative.at(i) = PETSC_TRUE;
                    lex_indices.at(i) = 0;
                } else {
                    negative.at(i) = PETSC_FALSE;
                }
            }
            CHKERRABORT(comm, AOApplicationToPetsc(lex2petsc, (PetscInt) lex_indices.n_elem, &lex_indices[0]));
            for (auto i{0}; i < negative.size(); ++i) {
                if (negative.at(i)) {
                    lex_indices(i) = -1;
                }
            }
            return lex_indices;
        }

        void FiniteStateSubset::State2Petsc(arma::Mat<PetscInt> state, PetscInt *indx) {
            cme::sub2ind_nd(fsp_size, state, indx);
            std::vector<PetscBool> negative(state.n_cols);
            for (auto i{0}; i < negative.size(); ++i) {
                if (indx[i] < 0) {
                    negative.at(i) = PETSC_TRUE;
                    indx[i] = 0;
                } else {
                    negative.at(i) = PETSC_FALSE;
                }
            }
            CHKERRABORT(comm, AOApplicationToPetsc(lex2petsc, (PetscInt) state.n_cols, indx));
            for (auto i{0}; i < negative.size(); ++i) {
                if (negative.at(i)) {
                    indx[i] = -1;
                }
            }
        }

        FiniteStateSubset::~FiniteStateSubset() {
            PetscMPIInt ierr;
            ierr = MPI_Comm_free(&comm);
            Zoltan_Destroy(&zoltan);
            CHKERRABORT(comm, ierr);
            Destroy();
        }

        std::tuple<PetscInt, PetscInt> FiniteStateSubset::GetLayoutStartEnd() {
            PetscInt start, end, ierr;
            ierr = PetscLayoutGetRange(vec_layout, &start, &end);
            CHKERRABORT(comm, ierr);
            return std::make_tuple(start, end);

        }

        arma::Row<PetscReal> FiniteStateSubset::SinkStatesReduce(Vec P) {
            PetscInt ierr;

            arma::Row<PetscReal> local_sinks(fsp_size.n_elem), global_sinks(fsp_size.n_elem);

            PetscInt p_local_size;
            ierr = VecGetLocalSize(P, &p_local_size);
            CHKERRABORT(comm, ierr);

            if (p_local_size != local_states.n_cols + fsp_size.n_elem) {
                printf("FiniteStateSubset::SinkStatesReduce: The layout of p and FiniteStateSubset do not match.\n");
                MPI_Abort(comm, 1);
            }

            PetscReal *p_data;
            VecGetArray(P, &p_data);
            for (auto i{0}; i < fsp_size.n_elem; ++i) {
                local_sinks(i) = p_data[p_local_size - 1 - i];
                ierr = MPI_Allreduce(&local_sinks[i], &global_sinks[i], 1, MPIU_REAL, MPI_SUM, comm);
                CHKERRABORT(comm, ierr);
            }

            return global_sinks;
        }

        MPI_Comm FiniteStateSubset::GetComm() {
            return comm;
        }

        arma::Row<PetscInt> FiniteStateSubset::GetFSPSize() {
            return fsp_size;
        }

        PetscInt FiniteStateSubset::GetNumLocalStates() {
            return n_local_states;
        }

        PetscInt FiniteStateSubset::GetNumSpecies() {
            return (PetscInt(fsp_size.n_elem));
        }

        void FiniteStateSubset::PrintAO() {
            CHKERRABORT(comm, AOView(lex2petsc, PETSC_VIEWER_STDOUT_(comm)));
        }

        arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species) {
            MPI_Comm comm;
            PetscObjectGetComm((PetscObject) P, &comm);

            PetscReal *local_data;
            VecGetArray(P, &local_data);

            arma::Col<PetscReal> p_local(local_data, fsp.n_local_states, false, true);
            arma::Col<PetscReal> v(fsp.fsp_size(species) + 1);
            v.fill(0.0);

            for (PetscInt i{0}; i < fsp.n_local_states; ++i) {
                v(fsp.local_states(species, i)) += p_local(i);
            }

            MPI_Barrier(comm);

            arma::Col<PetscReal> w(fsp.fsp_size(species) + 1);
            w.fill(0.0);
            MPI_Allreduce(&v[0], &w[0], v.size(), MPI_DOUBLE, MPI_SUM, comm);

            VecRestoreArray(P, &local_data);
            return w;
        }

        void FiniteStateSubset::Destroy() {
            if (lex2petsc) AODestroy(&lex2petsc);
            if (vec_layout) PetscLayoutDestroy(&vec_layout);
        }

        AO FiniteStateSubset::GetAO() {
            return lex2petsc;
        }

        PetscInt FiniteStateSubset::GetNumGlobalStates() {
            return n_states_global;
        }

        // Interface to Zoltan
        int zoltan_num_obj(void *data, int *ierr) {
            *ierr = 0;
            return ((FiniteStateSubset::AdjacencyData *) data)->num_local_states;
        }

        void zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries,
                             ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim,
                             float *obj_wgts, int *ierr) {
            auto adj_data = (FiniteStateSubset::AdjacencyData *) data;
            for (int i{0}; i < adj_data->num_local_states; ++i) {
                global_id[i] = adj_data->states_gid[i];
                local_id[i] = (ZOLTAN_ID_TYPE) i;
            }
            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_geom(void *data, int *ierr) {
            auto fss_data = (FiniteStateSubset*) data;
            *ierr = ZOLTAN_OK;
            return fss_data->geom_data.dim;
        }

        void
        zoltan_geom_multi(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids,
                          ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr) {
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            auto fss_data = (FiniteStateSubset*) data;
            ZOLTAN_ID_TYPE local_idx;
            for (auto i{0}; i < num_obj; ++i){
                local_idx = local_ids[i];
                for (auto j{0}; j < num_dim; ++j){
                    geom_vec[num_dim*i + j] = fss_data->geom_data.states_coo[num_dim*local_idx + j];
                }
            }
        }

        void FiniteStateSubset::GenerateGeomData(arma::Row<PetscInt> &fsp_size, arma::Mat<PetscInt> &local_states_tmp) {
            // Enter state coordinates of the three largest dimensions
            PetscInt n_local_tmp = local_states_tmp.n_cols;
            PetscInt dim = local_states_tmp.n_rows < 3 ? local_states_tmp.n_rows : 3;
            arma::uvec sorted_dims = arma::sort_index(fsp_size,"descend");
            geom_data.dim = dim;
            geom_data.states_coo = new double[dim*n_local_tmp];
            for (auto i {0}; i < n_local_tmp; ++i){
                for (auto j {0}; j < dim; ++j){
                    geom_data.states_coo[i*dim + j] = double(1.0*local_states_tmp(sorted_dims(j), i));
                }
            }
        }

        void FiniteStateSubset::FreeGeomData() {
            geom_data.dim = 0;
            delete[] geom_data.states_coo;
        }

        void FiniteStateSubset::GenerateGraphData(arma::Mat<PetscInt> &local_states_tmp) {
            PetscErrorCode ierr;

            auto n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            arma::Col<PetscInt> x((size_t) n_species); // vector to iterate through local states
            arma::Mat<PetscInt> rx((size_t) n_species, (size_t) stoichiometry.n_cols); // states reachable from x
            arma::Row<PetscInt> irx;

            adj_data.states_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(n_local_tmp * sizeof(ZOLTAN_ID_TYPE), __FILE__,
                                                                __LINE__);
            adj_data.num_edges = new int[n_local_tmp];

            adj_data.edge_ptr = new int[n_local_tmp + 1];

            adj_data.reachable_states = (ZOLTAN_ID_PTR) Zoltan_Malloc(
                    2*(1 + stoichiometry.n_cols) * n_local_tmp * sizeof(ZOLTAN_ID_TYPE),
                    __FILE__, __LINE__);

            adj_data.reachable_states_proc = new int[2 * n_local_tmp * (1 + stoichiometry.n_cols)];

            adj_data.num_local_states = n_local_tmp;

            adj_data.num_reachable_states = 0;
            for (auto i = 0; i < n_local_tmp; ++i) {
                x = local_states_tmp.col(i);
                sub2ind_nd<PetscInt, ZOLTAN_ID_TYPE>(fsp_size, x, &adj_data.states_gid[i]);

                adj_data.num_edges[i] = 0;
                // Find indices of states connected to x via a reaction
                rx = arma::join_horiz(repmat(x, 1, stoichiometry.n_cols) - stoichiometry, repmat(x, 1, stoichiometry.n_cols) + stoichiometry);
                irx = sub2ind_nd(fsp_size, rx);
                irx = unique(irx);
                adj_data.edge_ptr[i] = (int) adj_data.num_reachable_states;

                for (PetscInt j = 0; j < irx.n_elem; ++j) {
                    if (irx.at(j) >= 0) {
                        adj_data.reachable_states[adj_data.num_reachable_states] = (ZOLTAN_ID_TYPE) irx.at(
                                j);
                        adj_data.num_reachable_states++;
                        adj_data.num_edges[i]++;
                    }
                }
            }
            adj_data.edge_ptr[n_local_tmp] = adj_data.num_reachable_states;

            //
            // Renumbering the graph vertices for fast graph building
            //
            PetscMPIInt num_procs; // NOTE TO SELF: ALWAYS PAY ATTENTION TO INTEROPERABILITY BETWEEN MPI AND PETSC

            const PetscInt *procs_start;
            PetscLayout vec_layout_tmp;
            arma::Row<PetscInt> local_idxs = sub2ind_nd(fsp_size, local_states_tmp);
            arma::Row<PetscInt> reachable_states_idxs;

            ierr = AOCreateMemoryScalable(comm, n_local_tmp, &local_idxs[0], NULL, &adj_data.lex2zoltan);
            CHKERRABORT(comm, ierr);

            MPI_Comm_size(comm, &num_procs);
            ierr = PetscLayoutCreate(comm, &vec_layout_tmp);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(vec_layout_tmp, n_local_tmp);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout_tmp);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRanges(vec_layout_tmp, &procs_start);
            CHKERRABORT(comm, ierr);


            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_local_states, &local_idxs[0]);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < adj_data.num_local_states; ++i) {
                adj_data.states_gid[i] = (ZOLTAN_ID_TYPE) local_idxs[i];
            }

            reachable_states_idxs.set_size((size_t) adj_data.num_reachable_states);
            for (auto i{0}; i < adj_data.num_reachable_states; ++i) {
                reachable_states_idxs(i) = (PetscInt) adj_data.reachable_states[i];
            }
            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_reachable_states, &reachable_states_idxs[0]);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < adj_data.num_reachable_states; ++i) {
                adj_data.reachable_states[i] = (ZOLTAN_ID_TYPE) reachable_states_idxs(i);
            }

            // Figure out which processors own the neighbors, this requires communication via Petsc's AO
            for (auto i{0}; i < adj_data.num_reachable_states; ++i) {
                for (auto j{0}; j < num_procs; ++j) {
                    if ((procs_start[j] <= reachable_states_idxs[i]) &&
                        (procs_start[j + 1] > reachable_states_idxs[i])) {
                        adj_data.reachable_states_proc[i] = (int) j;
                    }
                }
            }

            PetscLayoutDestroy(&vec_layout_tmp);
        }

        void FiniteStateSubset::FreeGraphData() {
            delete[] adj_data.num_edges;
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.reachable_states, __FILE__, __LINE__);
            delete[] adj_data.edge_ptr;
            delete[] adj_data.reachable_states_proc;
            AODestroy(&adj_data.lex2zoltan);
        }

        void FiniteStateSubset::GenerateHyperGraphData(arma::Mat<PetscInt> &local_states_tmp) {
            PetscErrorCode ierr;

            auto n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            arma::Col<PetscInt> x((size_t) n_species); // vector to iterate through local states
            arma::Mat<PetscInt> rx((size_t) n_species, (size_t) stoichiometry.n_cols); // states reachable from x
            arma::Row<PetscInt> irx;

            adj_data.states_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(n_local_tmp * sizeof(ZOLTAN_ID_TYPE), __FILE__,
                                                                __LINE__);
            adj_data.num_edges = new int[n_local_tmp];

            adj_data.edge_ptr = new int[n_local_tmp + 1];

            adj_data.reachable_states = (ZOLTAN_ID_PTR) Zoltan_Malloc(
                    (1 + stoichiometry.n_cols) * n_local_tmp * sizeof(ZOLTAN_ID_TYPE),
                    __FILE__, __LINE__);

            adj_data.num_local_states = n_local_tmp;

            adj_data.num_reachable_states = 0;
            for (auto i = 0; i < n_local_tmp; ++i) {
                x = local_states_tmp.col(i);
                sub2ind_nd<PetscInt, ZOLTAN_ID_TYPE>(fsp_size, x, &adj_data.states_gid[i]);

                adj_data.num_edges[i] = 0;
                // Find indices of states connected to x on its row
                rx = repmat(x, 1, stoichiometry.n_cols) - stoichiometry;
                irx = sub2ind_nd(fsp_size, rx);
                irx = unique(irx);
                // Add diagonal entries
                adj_data.edge_ptr[i] = (int) adj_data.num_reachable_states;
                adj_data.reachable_states[adj_data.num_reachable_states] = adj_data.states_gid[i];
                adj_data.num_reachable_states++;
                adj_data.num_edges[i]++;
                // Add off-diagonal entries
                for (PetscInt j = 0; j < irx.n_elem; ++j) {
                    if (irx.at(j) >= 0) {
                        adj_data.reachable_states[adj_data.num_reachable_states] = (ZOLTAN_ID_TYPE) irx.at(
                                j);
                        adj_data.num_reachable_states++;
                        adj_data.num_edges[i]++;
                    }
                }
            }
            adj_data.edge_ptr[n_local_tmp] = adj_data.num_reachable_states;

            //
            // Renumbering the graph vertices for fast graph building
            //
            PetscMPIInt num_procs; // NOTE TO SELF: ALWAYS PAY ATTENTION TO INTEROPERABILITY BETWEEN MPI AND PETSC

            const PetscInt *procs_start;
            arma::Row<PetscInt> local_idxs = sub2ind_nd(fsp_size, local_states_tmp);
            arma::Row<PetscInt> reachable_states_idxs;

            ierr = AOCreateMemoryScalable(comm, n_local_tmp, &local_idxs[0], NULL, &adj_data.lex2zoltan);
            CHKERRABORT(comm, ierr);

            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_local_states, &local_idxs[0]);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < adj_data.num_local_states; ++i) {
                adj_data.states_gid[i] = (ZOLTAN_ID_TYPE) local_idxs[i];
            }

            reachable_states_idxs.set_size((size_t) adj_data.num_reachable_states);
            for (auto i{0}; i < adj_data.num_reachable_states; ++i) {
                reachable_states_idxs(i) = (PetscInt) adj_data.reachable_states[i];
            }
            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_reachable_states, &reachable_states_idxs[0]);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < adj_data.num_reachable_states; ++i) {
                adj_data.reachable_states[i] = (ZOLTAN_ID_TYPE) reachable_states_idxs(i);
            }
        }

        void FiniteStateSubset::FreeHyperGraphData() {
            delete[] adj_data.num_edges;
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.reachable_states, __FILE__, __LINE__);
            delete[] adj_data.edge_ptr;
        }

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
            auto *hg_data = (FiniteStateSubset::AdjacencyData *) data;
            *num_lists = hg_data->num_local_states;
            *num_pins = hg_data->num_reachable_states;
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                   ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr) {
            auto hg_data = (FiniteStateSubset::AdjacencyData *) data;

            if ((num_vertices != hg_data->num_local_states) || (num_pins != hg_data->num_reachable_states) ||
                (format != ZOLTAN_COMPRESSED_VERTEX)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }

            for (int i{0}; i < num_vertices; ++i) {
                vtx_gid[i] = hg_data->states_gid[i];
                vtx_edge_ptr[i] = hg_data->edge_ptr[i];
            }

            for (int i{0}; i < num_pins; ++i) {
                pin_gid[i] = hg_data->reachable_states[i];
            }
            *ierr = ZOLTAN_OK;
        }

        int zoltan_num_edges(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                             ZOLTAN_ID_PTR local_id, int *ierr) {
            auto g_data = (FiniteStateSubset::AdjacencyData *) data;
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return -1;
            }
            ZOLTAN_ID_TYPE indx = *local_id;
            *ierr = ZOLTAN_OK;
            return g_data->num_edges[indx];
        }

        void zoltan_edge_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                              ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim,
                              float *ewgts, int *ierr) {
            auto g_data = (FiniteStateSubset::AdjacencyData *) data;
            if ((num_gid_entries != 1) || (num_lid_entries != 1)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }
            ZOLTAN_ID_TYPE indx = *local_id;
            int k = 0;
            for (auto i = g_data->edge_ptr[indx]; i < g_data->edge_ptr[indx + 1]; ++i) {
                nbor_global_id[k] = g_data->reachable_states[i];
                nbor_procs[k] = g_data->reachable_states_proc[i];
                k++;
            }
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubset::CallZoltanLoadBalancing() {
            zoltan_err = Zoltan_LB_Partition(
                    zoltan,
                    &changes,
                    &num_gid_entries,
                    &num_lid_entries,
                    &num_import,
                    &import_global_ids,
                    &import_local_ids,
                    &import_procs,
                    &import_to_part,
                    &num_export,
                    &export_global_ids,
                    &export_local_ids,
                    &export_procs,
                    &export_to_part);
            CHKERRABORT(comm, zoltan_err);
        }

        void FiniteStateSubset::FreeZoltanParts() {
            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
        }

        void FiniteStateSubset::LocalStatesFromAO() {
            PetscErrorCode ierr;

            std::vector<PetscInt> petsc_local_indices((size_t) n_local_states);
            ierr = PetscLayoutGetRange(vec_layout, &petsc_local_indices[0], NULL);
            CHKERRABORT(comm, ierr);
            for (PetscInt i{0}; i < n_local_states; ++i) {
                petsc_local_indices[i] = petsc_local_indices[0] + i;
            }
            ierr = AOPetscToApplication(lex2petsc, n_local_states, &petsc_local_indices[0]);
            CHKERRABORT(comm, ierr);

            local_states = ind2sub_nd(fsp_size, petsc_local_indices);
        }

        arma::Mat<PetscInt> FiniteStateSubset::get_my_naive_local_states() {
            PetscErrorCode ierr;

            PetscInt local_start_tmp, local_end_tmp, n_local_tmp;
            arma::Row<PetscInt> range_tmp;
            PetscLayout vec_layout_tmp;
            ierr = PetscLayoutCreate(comm, &vec_layout_tmp);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetSize(vec_layout_tmp, n_states_global);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout_tmp);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRange(vec_layout_tmp, &local_start_tmp, &local_end_tmp);
            CHKERRABORT(comm, ierr);
            n_local_tmp = local_end_tmp - local_start_tmp;
            range_tmp.set_size(n_local_tmp);
            for (auto i{0}; i < n_local_tmp; ++i) {
                range_tmp(i) = local_start_tmp + i;
            }
            ierr = PetscLayoutDestroy(&vec_layout_tmp);
            CHKERRABORT(comm, ierr);

            return ind2sub_nd(fsp_size, range_tmp);
        }

        void FiniteStateSubset::ComputePetscOrderingFromZoltan() {
            PetscErrorCode ierr;

            PetscInt n_local_tmp = adj_data.num_local_states;
            //
            // Copy Zoltan's result to Petsc IS
            //
            IS processor_id, global_numbering;
            PetscInt *export_procs_petsc = new PetscInt[n_local_tmp];
            for (auto i{0}; i < n_local_tmp; ++i){
                export_procs_petsc[i] = (PetscInt) export_procs[i];
            }
            ierr = ISCreateGeneral(comm, (PetscInt) num_export, export_procs_petsc, PETSC_COPY_VALUES, &processor_id);
            CHKERRABORT(comm, ierr);
            ierr = ISPartitioningToNumbering(processor_id, &global_numbering);
            CHKERRABORT(comm, ierr);

            //
            // Generate layout
            //
            PetscInt layout_start, layout_end;

            // Figure out how many states each processor own by reducing the proc_id_array
            PetscMPIInt my_rank, comm_size;
            MPI_Comm_size(comm, &comm_size);
            MPI_Comm_rank(comm, &my_rank);
            std::vector<PetscInt> n_own((size_t) comm_size);
            ierr = ISPartitioningCount(processor_id, comm_size, &n_own[0]);
            CHKERRABORT(comm, ierr);
            n_local_states = n_own[my_rank];

            ierr = PetscLayoutCreate(comm, &vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(vec_layout, n_local_states + n_species);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetSize(vec_layout, PETSC_DECIDE);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRange(vec_layout, &layout_start, &layout_end);
            CHKERRABORT(comm, ierr);

            //
            // Create the AO object that maps from lexicographic ordering to Petsc Vec ordering
            //

            // Adjust global numbering to add sink states
            const PetscInt *proc_id_array, *global_numbering_array;
            ierr = ISGetIndices(processor_id, &proc_id_array);
            CHKERRABORT(comm, ierr);
            ierr = ISGetIndices(global_numbering, &global_numbering_array);
            CHKERRABORT(comm, ierr);
            std::vector<PetscInt> corrected_global_numbering(n_local_tmp + n_species);
            for (PetscInt i{0}; i < n_local_tmp; ++i) {
                corrected_global_numbering.at(i) =
                        global_numbering_array[i] + proc_id_array[i] * ((PetscInt) fsp_size.n_elem);
            }

            std::vector<PetscInt> fsp_indices(n_local_tmp + n_species);
            for (PetscInt i = 0; i < n_local_tmp; ++i) {
                fsp_indices.at(i) = (PetscInt) adj_data.states_gid[i];
            }
            AOPetscToApplication(adj_data.lex2zoltan, n_local_tmp, &fsp_indices[0]);

            for (PetscInt i = 0; i < n_species; ++i) {
                corrected_global_numbering[n_local_tmp + n_species - i - 1] = layout_end - 1 - i;
                fsp_indices[n_local_tmp + n_species - i - 1] =
                        n_states_global + my_rank * n_species + i;
            }
            ierr = ISRestoreIndices(processor_id, &proc_id_array);
            CHKERRABORT(comm, ierr);
            ierr = ISRestoreIndices(global_numbering, &global_numbering_array);
            CHKERRABORT(comm, ierr);
            ierr = ISDestroy(&processor_id);
            CHKERRABORT(comm, ierr);
            ierr = ISDestroy(&global_numbering);
            CHKERRABORT(comm, ierr);

            IS fsp_is, petsc_is;
            ierr = ISCreateGeneral(comm, n_local_tmp + n_species, &fsp_indices[0],
                                   PETSC_COPY_VALUES, &fsp_is);
            CHKERRABORT(comm, ierr);
            ierr = ISCreateGeneral(comm, n_local_tmp + n_species, &corrected_global_numbering[0],
                                   PETSC_COPY_VALUES, &petsc_is);
            CHKERRABORT(comm, ierr);
            ierr = AOCreate(comm, &lex2petsc);
            CHKERRABORT(comm, ierr);
            ierr = AOSetIS(lex2petsc, fsp_is, petsc_is);
            CHKERRABORT(comm, ierr);
            ierr = AOSetType(lex2petsc, AOMEMORYSCALABLE);
            CHKERRABORT(comm, ierr);
            ierr = AOSetFromOptions(lex2petsc);
            CHKERRABORT(comm, ierr);

            ierr = ISDestroy(&fsp_is);
            CHKERRABORT(comm, ierr);
            ierr = ISDestroy(&petsc_is);
            CHKERRABORT(comm, ierr);
        }

        std::string part2str(PartitioningType part) {
            switch (part) {
                case Naive:
                    return std::string("naive");
                case RCB:
                    return std::string("rcb");
                case Graph:
                    return std::string("graph");
                case HyperGraph:
                    return std::string("hyper_graph");
                case Hierarch:
                    return std::string("hierarch");
                default:
                    return std::string("naive");
            }
        }

        PartitioningType str2part(std::string str) {
            if (str == "naive" || str == "Naive" || str == "NAIVE") {
                return Naive;
            } else if (str == "rcb" || str == "RCB" || str == "RcB" || str == "Geometric" || str == "coo") {
                return RCB;
            } else if (str == "graph" || str == "Graph" || str == "GRAPH") {
                return Graph;
            } else if (str == "hierarch" || str == "Hierarch" || str == "HIERARCH" || str == "artantis" || str == "Artantis") {
                return Hierarch;
            } else {
                return HyperGraph;
            }
        }
    }
}
