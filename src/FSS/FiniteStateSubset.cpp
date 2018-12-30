//
// Created by Huy Vo on 12/4/18.
//
#include <FSS/FiniteStateSubset.h>
#include "FiniteStateSubset.h"


namespace cme {
    namespace parallel {
        FiniteStateSubset::FiniteStateSubset(MPI_Comm new_comm) {
            PetscErrorCode ierr;
            MPI_Comm_dup(new_comm, &comm);
            partitioning_type = NotSet;
            local_states.resize(0);
            fsp_size.resize(0);
            n_states_global = 0;
            stoichiometry.resize(0);

            /// Set up Zoltan
            zoltan = Zoltan_Create(comm);
            // This imbalance tolerance is universal for all methods
            Zoltan_Set_Param(zoltan, "IMBALANCE_TOL", "1.1");

            // Register query functions to zoltan
            Zoltan_Set_Num_Obj_Fn(zoltan, &zoltan_num_obj, (void *) &this->adj_data);
            Zoltan_Set_Obj_List_Fn(zoltan, &zoltan_obj_list, (void *) &this->adj_data);
            Zoltan_Set_Num_Geom_Fn(zoltan, &zoltan_num_geom, (void *) this);
            Zoltan_Set_Geom_Multi_Fn(zoltan, &zoltan_geom_multi, (void *) this);
            Zoltan_Set_Num_Edges_Fn(zoltan, &zoltan_num_edges, (void *) &this->adj_data);
            Zoltan_Set_Edge_List_Fn(zoltan, &zoltan_edge_list, (void *) &this->adj_data);
            Zoltan_Set_HG_Size_CS_Fn(zoltan, &zoltan_get_hypergraph_size, (void *) &this->adj_data);
            Zoltan_Set_HG_CS_Fn(zoltan, &zoltan_get_hypergraph, (void *) &this->adj_data);

            /// Register event logging
            ierr = PetscLogEventRegister("Generate graph data", 0, &generate_graph_data);
            CHKERRABORT(comm, ierr);
            ierr = PetscLogEventRegister("Zoltan partitioning", 0, &call_partitioner);
            CHKERRABORT(comm, ierr);
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

        void FiniteStateSubset::GenerateGeomData(arma::Row<PetscInt> &fsp_size, arma::Mat<PetscInt> &local_states_tmp) {
            // Enter state coordinates of the three largest dimensions
            PetscInt n_local_tmp = local_states_tmp.n_cols;
            PetscInt dim = local_states_tmp.n_rows < 3 ? local_states_tmp.n_rows : 3;
            arma::uvec sorted_dims = arma::sort_index(fsp_size, "descend");
            geom_data.dim = dim;
            geom_data.states_coo = new double[dim * n_local_tmp];
            for (auto i{0}; i < n_local_tmp; ++i) {
                for (auto j{0}; j < dim; ++j) {
                    geom_data.states_coo[i * dim + j] = double(1.0 * local_states_tmp(sorted_dims(j), i));
                }
            }
        }

        void FiniteStateSubset::FreeGeomData() {
            geom_data.dim = 0;
            delete[] geom_data.states_coo;
        }

        void FiniteStateSubset::GenerateVertexData(arma::Mat<PetscInt> &local_states_tmp) {
            PetscErrorCode ierr;

            ierr = PetscLogEventBegin(generate_graph_data, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);

            auto n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            arma::Col<PetscInt> x((size_t) n_species); // vector to iterate through local states

            adj_data.states_gid = new PetscInt[n_local_tmp];

            adj_data.states_weights = new float[n_local_tmp];

            adj_data.num_local_states = n_local_tmp;

            for (auto i = 0; i < n_local_tmp; ++i) {
                x = local_states_tmp.col(i);
                sub2ind_nd<PetscInt, PetscInt>(fsp_size, x, &adj_data.states_gid[i]);

                // Enter vertex's weight
                adj_data.states_weights[i] =
                        (float) 1.0f;
            }

            //
            // Renumbering the graph vertices for fast graph building
            //
            ierr = AOCreateMemoryScalable(comm, n_local_tmp, &adj_data.states_gid[0], NULL, &adj_data.lex2zoltan);
            CHKERRABORT(comm, ierr);

            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_local_states, &adj_data.states_gid[0]);
            CHKERRABORT(comm, ierr);

            ierr = PetscLogEventEnd(generate_graph_data, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);
        }

        void FiniteStateSubset::FreeVertexData() {
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            AODestroy(&adj_data.lex2zoltan);
        }

        void FiniteStateSubset::GenerateGraphData(arma::Mat<PetscInt> &local_states_tmp) {
            PetscErrorCode ierr;

            ierr = PetscLogEventBegin(generate_graph_data, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);

            auto n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            arma::Col<PetscInt> x((size_t) n_species); // vector to iterate through local states
            arma::Mat<PetscInt> rx((size_t) n_species, (size_t) 2*stoichiometry.n_cols); // states reachable from x
            arma::Row<PetscInt> irx(2*stoichiometry.n_cols), irx1(stoichiometry.n_cols), irx2(stoichiometry.n_cols), cc;
            arma::Row<float> wrx(2*stoichiometry.n_cols);
            arma::uvec ia(stoichiometry.n_cols), ib(stoichiometry.n_cols), ic, id;

            adj_data.states_gid = new PetscInt[n_local_tmp];

            adj_data.states_weights = new float[n_local_tmp];

            adj_data.num_edges = new int[n_local_tmp];

            adj_data.edge_ptr = new int[n_local_tmp + 1];

            adj_data.reachable_states = new PetscInt[2 * n_local_tmp * (1 + stoichiometry.n_cols)];

            adj_data.reachable_states_proc = new int[2 * n_local_tmp * (1 + stoichiometry.n_cols)];

            adj_data.edge_weights = new float[2 * n_local_tmp * (1 + stoichiometry.n_cols)];

            adj_data.num_local_states = n_local_tmp;

            adj_data.num_reachable_states = 0;
            for (auto i = 0; i < n_local_tmp; ++i) {
                x = local_states_tmp.col(i);
                sub2ind_nd<PetscInt, PetscInt>(fsp_size, x, &adj_data.states_gid[i]);

                adj_data.num_edges[i] = 0;
                // Find indices of states connected to x via a reaction
                for (auto j{0}; j < stoichiometry.n_cols; ++j){
                    rx.col(j) = x - stoichiometry.col(j);
                    rx.col(stoichiometry.n_cols + j) = x + stoichiometry.col(j);
                }
                sub2ind_nd(fsp_size, rx, &irx[0]);

                irx1 = irx(
                        arma::span(0,
                                   stoichiometry.n_cols - 1)); // indices of nonzero entries on row x of the FSP matrix
                irx2 = irx(arma::span(stoichiometry.n_cols, 2 * stoichiometry.n_cols - 1));

                // Compute edge weights
                wrx.fill(0.0f);
                ib = find_unique(irx);
                arma::intersect(cc, ic, id, irx.elem(ib).t(), irx1);
                wrx.elem(ib.elem(ic)) += 1.0f;
                arma::intersect(cc, ic, id, irx.elem(ib).t(), irx2);
                wrx.elem(ib.elem(ic)) += 1.0f;

                // Enter vertex's weight
                ia = find_unique(irx1);
                adj_data.states_weights[i] =
                        (float) 2.0f * stoichiometry.n_cols + 1.0f * stoichiometry.n_cols + 1.0f * ia.n_elem;

                // Enter edges and their weights
                adj_data.edge_ptr[i] = (int) adj_data.num_reachable_states;
                for (PetscInt j = 0; j < ib.n_elem; ++j) {
                    if (irx.at(ib(j)) >= 0) {
                        adj_data.reachable_states[adj_data.num_reachable_states] = irx.at(
                                ib(j));
                        adj_data.edge_weights[adj_data.num_reachable_states] = (float) wrx.at(ib(j));
                        adj_data.num_reachable_states++;
                        adj_data.num_edges[i]++;
                    }
                }
            }
            adj_data.edge_ptr[n_local_tmp] = adj_data.num_reachable_states;

            //
            // Renumbering the graph vertices for fast graph building
            //
            ierr = AOCreateMemoryScalable(comm, n_local_tmp, &adj_data.states_gid[0], NULL, &adj_data.lex2zoltan);
            CHKERRABORT(comm, ierr);

            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_local_states, &adj_data.states_gid[0]);
            CHKERRABORT(comm, ierr);

            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_reachable_states,
                                        &adj_data.reachable_states[0]);
            CHKERRABORT(comm, ierr);

            ierr = PetscLogEventEnd(generate_graph_data, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);
        }

        void FiniteStateSubset::FreeGraphData() {
            delete[] adj_data.num_edges;
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.reachable_states, __FILE__, __LINE__);
            delete[] adj_data.reachable_states_proc;
            delete[] adj_data.edge_ptr;
            delete[] adj_data.states_weights;
            delete[] adj_data.edge_weights;
            AODestroy(&adj_data.lex2zoltan);
        }

        void FiniteStateSubset::GenerateHyperGraphData(arma::Mat<PetscInt> &local_states_tmp) {
            PetscErrorCode ierr;

            ierr = PetscLogEventBegin(generate_graph_data, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);

            auto n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            arma::Col<PetscInt> x((size_t) n_species); // vector to iterate through local states
            arma::Mat<PetscInt> rx((size_t) n_species, (size_t) stoichiometry.n_cols); // states reachable from x
            arma::Row<PetscInt> irx;

            adj_data.states_gid = new PetscInt[n_local_tmp];

            adj_data.states_weights = new float[n_local_tmp];

            adj_data.num_edges = new int[n_local_tmp];

            adj_data.edge_ptr = new int[n_local_tmp + 1];

            adj_data.reachable_states = new PetscInt[n_local_tmp * (stoichiometry.n_cols + 1)];

            adj_data.num_local_states = n_local_tmp;

            adj_data.num_reachable_states = 0;
            for (auto i = 0; i < n_local_tmp; ++i) {
                x = local_states_tmp.col(i);
                sub2ind_nd<PetscInt, PetscInt>(fsp_size, x, &adj_data.states_gid[i]);

                adj_data.num_edges[i] = 0;
                // Find indices of states connected to x on its row
                rx = repmat(x, 1, stoichiometry.n_cols) - stoichiometry;
                irx = sub2ind_nd(fsp_size, rx);
                irx = unique(irx);
                // Compute vertex weight
                adj_data.states_weights[i] =
                        (float) 2.0f * stoichiometry.n_cols + 1.0f * stoichiometry.n_cols + 1.0f * irx.n_elem;

                // Add diagonal pin
                adj_data.edge_ptr[i] = (int) adj_data.num_reachable_states;
                adj_data.reachable_states[adj_data.num_reachable_states] = adj_data.states_gid[i];
                adj_data.num_reachable_states++;
                adj_data.num_edges[i]++;
                // Add off-diagonal pins
                for (PetscInt j = 0; j < irx.n_elem; ++j) {
                    if (irx.at(j) >= 0) {
                        adj_data.reachable_states[adj_data.num_reachable_states] = irx.at(
                                j);
                        adj_data.num_reachable_states++;
                        adj_data.num_edges[i]++;
                    }
                }
            }
            adj_data.edge_ptr[n_local_tmp] = adj_data.num_reachable_states;

            //
            // Renumbering the hypergraph vertices for fast hypergraph building
            //
            ierr = AOCreateMemoryScalable(comm, n_local_tmp, &adj_data.states_gid[0], NULL, &adj_data.lex2zoltan);
            CHKERRABORT(comm, ierr);

            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_local_states, &adj_data.states_gid[0]);
            CHKERRABORT(comm, ierr);

            ierr = AOApplicationToPetsc(adj_data.lex2zoltan, adj_data.num_reachable_states,
                                        &adj_data.reachable_states[0]);
            CHKERRABORT(comm, ierr);

            ierr = PetscLogEventEnd(generate_graph_data, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);
        }

        void FiniteStateSubset::FreeHyperGraphData() {
            delete[] adj_data.num_edges;
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.reachable_states, __FILE__, __LINE__);
            delete[] adj_data.edge_ptr;
            delete[] adj_data.states_weights;
            AODestroy(&adj_data.lex2zoltan);
        }

        void FiniteStateSubset::CallZoltanLoadBalancing() {
            PetscErrorCode ierr;

            ierr = PetscLogEventBegin(call_partitioner, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);

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

            ierr = PetscLogEventEnd(call_partitioner, 0, 0, 0, 0);
            CHKERRABORT(comm, ierr);
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

        arma::Mat<PetscInt> FiniteStateSubset::compute_my_naive_local_states() {
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
            for (auto i{0}; i < n_local_tmp; ++i) {
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

        void FiniteStateSubset::SetRepartApproach(PartitioningApproach approach) {
            repart_approach = approach;
            switch (approach) {
                case FromScratch:
                    zoltan_part_opt = "PARTITION";
                    break;
                case Repartition:
                    zoltan_part_opt = "REPARTITION";
                    break;
                case Refine:
                    zoltan_part_opt = "REFINE";
                    break;
                default:
                    break;
            }
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
            } else if (str == "hierarch" || str == "Hierarch" || str == "HIERARCH" || str == "artantis" ||
                       str == "Artantis") {
                return Hierarch;
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
                case Refine:
                    return std::string("refine");
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
            } else if (str == "refine" || str == "Refine") {
                return Refine;
            } else {
                return FromScratch;
            }
        }
    }
}
