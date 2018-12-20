//
// Created by Huy Vo on 12/4/18.
//
#include <FiniteStateSubset.h>

#include "FiniteStateSubset.h"

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

            Zoltan_Set_Num_Obj_Fn(zoltan, &zoltan_num_obj, (void *) &this->adj_data);
            Zoltan_Set_Obj_List_Fn(zoltan, &zoltan_obj_list, (void *) &this->adj_data);
            Zoltan_Set_Num_Edges_Fn(zoltan, &zoltan_num_edges, (void *) &this->adj_data);
            Zoltan_Set_Edge_List_Fn(zoltan, &zoltan_edge_list, (void *) &this->adj_data);
            Zoltan_Set_HG_Size_CS_Fn(zoltan, &zoltan_get_hypergraph_size, (void *) &this->adj_data);
            Zoltan_Set_HG_CS_Fn(zoltan, &zoltan_get_hypergraph, (void *) &this->adj_data);
        };

        void FiniteStateSubset::SetStoichiometry(arma::Mat<PetscInt> SM) {
            stoichiometry = SM;
            stoich_set = 1;
        }

        void FiniteStateSubset::SetSize(arma::Row<PetscInt> &new_fsp_size) {
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

        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
            auto *hg_data = (FiniteStateSubset::AdjacencyData *) data;
            *num_lists = hg_data->num_local_states;
            *num_pins = hg_data->num_reachable_states_rows;
            *format = ZOLTAN_COMPRESSED_VERTEX;
            *ierr = ZOLTAN_OK;
        }

        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                   ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr) {
            auto hg_data = (FiniteStateSubset::AdjacencyData *) data;

            if ((num_vertices != hg_data->num_local_states) || (num_pins != hg_data->num_reachable_states_rows) ||
                (format != ZOLTAN_COMPRESSED_VERTEX)) {
                *ierr = ZOLTAN_FATAL;
                return;
            }

            for (int i{0}; i < num_vertices; ++i) {
                vtx_gid[i] = hg_data->states_gid[i];
                vtx_edge_ptr[i] = hg_data->rows_edge_ptr[i];
            }

            for (int i{0}; i < num_pins; ++i) {
                pin_gid[i] = hg_data->reachable_states_rows_gid[i];
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
            for (auto i = g_data->rows_edge_ptr[indx]; i < g_data->rows_edge_ptr[indx + 1]; ++i) {
                nbor_global_id[k] = g_data->reachable_states_rows_gid[i];
                nbor_procs[k] = g_data->reachable_states_rows_proc[i];
                k++;
            }
            for (auto i = g_data->cols_edge_ptr[indx]; i < g_data->cols_edge_ptr[indx + 1]; ++i) {
                nbor_global_id[k] = g_data->reachable_states_cols_gid[i];
                nbor_procs[k] = g_data->reachable_states_cols_proc[i];
                k++;
            }
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubset::call_zoltan_partitioner() {
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

        void FiniteStateSubset::free_zoltan_part_variables() {
            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
        }

        void FiniteStateSubset::get_local_states_from_ao() {
            PetscErrorCode ierr;

            std::vector<PetscInt> petsc_local_indices((size_t) n_local_states);
            ierr = PetscLayoutGetRange(vec_layout, &petsc_local_indices[0], NULL);
            CHKERRABORT(comm, ierr);
            for (PetscInt i{0}; i < n_local_states; ++i) {
                petsc_local_indices[i] = petsc_local_indices[0] + i;
            }
            ierr = AOPetscToApplication(lex2petsc, n_local_states, &petsc_local_indices[0]);
            CHKERRABORT(comm, ierr);

            local_states = ind2sub_nd<arma::Mat<PetscInt>>(fsp_size, petsc_local_indices);
        }

        arma::Mat<PetscInt> FiniteStateSubset::get_my_naive_local_states() {
            PetscErrorCode ierr;

            PetscInt local_start_tmp, local_end_tmp, n_local_tmp;
            arma::Row<PetscInt> range_tmp;
            ierr = PetscLayoutCreate(comm, &vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetSize(vec_layout, n_states_global);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRange(vec_layout, &local_start_tmp, &local_end_tmp);
            CHKERRABORT(comm, ierr);
            n_local_tmp = local_end_tmp - local_start_tmp;
            range_tmp.set_size(n_local_tmp);
            for (auto i{0}; i < n_local_tmp; ++i) {
                range_tmp(i) = local_start_tmp + i;
            }
            ierr = PetscLayoutDestroy(&vec_layout);
            CHKERRABORT(comm, ierr);

            return ind2sub_nd(fsp_size, range_tmp);
        }

        void FiniteStateSubset::compute_petsc_ordering_from_zoltan() {
            PetscErrorCode ierr;

            PetscInt n_local_tmp = adj_data.num_local_states;
            //
            // Copy Zoltan's result to Petsc IS
            //
            IS processor_id, global_numbering;
            ierr = ISCreateGeneral(comm, (PetscInt) num_export, export_procs, PETSC_COPY_VALUES, &processor_id);
            CHKERRABORT(comm, ierr);
            ierr = ISPartitioningToNumbering(processor_id, &global_numbering);
            CHKERRABORT(comm, ierr);

            ierr = PetscLayoutDestroy(&vec_layout);
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
            CHKERRABORT(comm, ISGetIndices(processor_id, &proc_id_array));
            CHKERRABORT(comm, ISGetIndices(global_numbering, &global_numbering_array));
            std::vector<PetscInt> corrected_global_numbering(n_local_tmp + n_species);
            for (PetscInt i{0}; i < n_local_tmp; ++i) {
                corrected_global_numbering.at(i) =
                        global_numbering_array[i] + proc_id_array[i] * ((PetscInt) fsp_size.n_elem);
            }

            std::vector<PetscInt> fsp_indices(n_local_tmp + n_species);
            for (PetscInt i = 0; i < n_local_tmp; ++i) {
                fsp_indices.at(i) = adj_data.states_gid[i];
            }

            for (PetscInt i = 0; i < n_species; ++i) {
                corrected_global_numbering[n_local_tmp + n_species - i - 1] = layout_end - 1 - i;
                fsp_indices[n_local_tmp + n_species - i - 1] =
                        n_states_global + my_rank * n_species + i;
            }
            CHKERRABORT(comm, ISRestoreIndices(processor_id, &proc_id_array));
            CHKERRABORT(comm, ISRestoreIndices(global_numbering, &global_numbering_array));
            CHKERRABORT(comm, ISDestroy(&processor_id));
            CHKERRABORT(comm, ISDestroy(&global_numbering));

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
                case Graph:
                    return std::string("graph");
                case HyperGraph:
                    return std::string("hyper_graph");
                default:
                    return std::string("naive");
            }
        }

        PartitioningType str2part(std::string str) {
            if (str == "naive" || str == "Naive" || str == "NAIVE") {
                return Naive;
            } else if (str == "graph" || str == "Graph" || str == "GRAPH") {
                return Graph;
            } else {
                return HyperGraph;
            }
        }

    }
}
