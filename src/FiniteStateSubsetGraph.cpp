//
// Created by Huy Vo on 12/4/18.
//

#include <FiniteStateSubsetGraph.h>

#include "FiniteStateSubsetGraph.h"

namespace cme {
    namespace parallel {
        FiniteStateSubsetGraph::FiniteStateSubsetGraph(MPI_Comm new_comm) : FiniteStateSubset(new_comm) {
            partitioning_type = Graph;
            Zoltan_Set_Param(zoltan, "LB_METHOD", "GRAPH");
            Zoltan_Set_Param(zoltan, "GRAPH_PACKAGE", "Parmetis");
            Zoltan_Set_Param(zoltan, "PARMETIS_METHOD", "PartKway");
            Zoltan_Set_Param(zoltan, "RETURN_LISTS", "PARTS");
            Zoltan_Set_Param(zoltan, "DEBUG_LEVEL", "0");
            Zoltan_Set_Param(zoltan, "OBJ_WEIGHT_DIM", "0"); // use Zoltan default vertex weights
            Zoltan_Set_Param(zoltan, "EDGE_WEIGHT_DIM", "0");// use Zoltan default hyperedge weights
            Zoltan_Set_Param(zoltan, "CHECK_GRAPH", "0");
        }

        void FiniteStateSubsetGraph::GenerateStatesAndOrdering() {
            PetscErrorCode ierr;
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0) {
                throw std::runtime_error("FiniteStateSubset: stoichiometry is required for Graph partitioning type.");
            }

            Zoltan_Set_Param(zoltan, "LB_APPROACH", "PARTITION");

            //
            // Initial temporary partitioning based on lexicographic ordering
            //
            arma::Mat<PetscInt> local_states_tmp = get_my_naive_local_states();

            //
            // Generate Graph data
            //
            generate_graph_data(local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with Petsc's IS
            //
            call_zoltan_partitioner();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            compute_petsc_ordering_from_zoltan();

            // Generate local states
            get_local_states_from_ao();

            free_graph_data();
            free_zoltan_part_variables();
        }

        void FiniteStateSubsetGraph::ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) {
            PetscErrorCode ierr;
            assert(new_fsp_size.n_elem == fsp_size.n_elem);
            for (auto i{0}; i < fsp_size.n_elem; ++i) {
                assert(new_fsp_size(i) >= fsp_size(i));
            }
            if (local_states.n_elem == 0) {
                PetscPrintf(comm,
                            "FiniteStateSubsetGraph: found empty local states array, probably because GenerateStatesAndOrdering was never called.\n");
                MPI_Abort(comm, -1);
            }
            if (lex2petsc) AODestroy(&lex2petsc);
            if (vec_layout) PetscLayoutDestroy(&vec_layout);

            arma::Row<PetscInt> fsp_size_old = fsp_size;
            SetSize(new_fsp_size);

            //
            // Switch Zoltan to Refine mode
            //
            Zoltan_Set_Param(zoltan, "LB_APPROACH", "REFINE"); // Migration is so cheap we don't need repartition

            //
            // Explore for new states that satisfy the new FSP bounds
            //

            PetscInt n_new_states;
            arma::Mat<PetscInt> new_candidates, local_states_tmp;
            arma::Row<PetscInt> is_new_states;

            new_candidates = get_my_naive_local_states();
            is_new_states.set_size(new_candidates.n_cols);

            n_new_states = 0;
            for (auto i{0}; i < new_candidates.n_cols; ++i) {
                is_new_states(i) = 0;
                for (auto j{0}; j < n_species; ++j) {
                    if (new_candidates(j, i) > fsp_size_old(j)) {
                        is_new_states(i) = 1;
                        n_new_states += 1;
                        break;
                    }
                }
            }
            PetscInt n_local_tmp = local_states.n_cols + n_new_states;
            local_states_tmp.set_size(fsp_size.n_elem, n_local_tmp);
            for (auto i{0}; i < local_states.n_cols; ++i) {
                local_states_tmp.col(i) = local_states.col(i);
            }
            for (auto i{0}, k{(PetscInt) local_states.n_cols}; i < new_candidates.n_cols; ++i) {
                if (is_new_states(i)) {
                    local_states_tmp.col(k) = new_candidates.col(i);
                    k++;
                }
            }

            //
            // Create the graph data
            //
            generate_graph_data(local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            //
            call_zoltan_partitioner();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            compute_petsc_ordering_from_zoltan();

            //
            // Generate local states
            //
            get_local_states_from_ao();

            free_graph_data();
            free_zoltan_part_variables();
        }

        void FiniteStateSubsetGraph::generate_graph_data(arma::Mat<PetscInt> &local_states_tmp) {
            PetscErrorCode ierr;

            PetscInt n_local_tmp = local_states_tmp.n_cols;

            arma::Col<PetscInt> x((size_t) n_species); // vector to iterate through local states
            arma::Mat<PetscInt> rx((size_t) n_species, (size_t) stoichiometry.n_cols); // states reachable from x
            arma::Row<PetscInt> irx;

            adj_data.states_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(n_local_tmp * sizeof(ZOLTAN_ID_TYPE), __FILE__,
                                                                __LINE__);
            adj_data.rows_edge_ptr = new int[n_local_tmp + 1];
            adj_data.cols_edge_ptr = new int[n_local_tmp + 1];
            adj_data.num_edges = new int[n_local_tmp];
            adj_data.reachable_states_rows_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(
                    (1 + stoichiometry.n_cols) * n_local_tmp * sizeof(ZOLTAN_ID_TYPE),
                    __FILE__, __LINE__);
            adj_data.reachable_states_rows_proc = new int[n_local_tmp * (1 + stoichiometry.n_cols)];
            adj_data.reachable_states_cols_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(
                    (1 + stoichiometry.n_cols) * n_local_tmp * sizeof(ZOLTAN_ID_TYPE),
                    __FILE__, __LINE__);
            adj_data.reachable_states_cols_proc = new int[n_local_tmp * (1 + stoichiometry.n_cols)];


            adj_data.num_local_states = n_local_tmp;
            adj_data.num_reachable_states_rows = 0;
            adj_data.num_reachable_states_cols = 0;
            for (auto i = 0; i < n_local_tmp; ++i) {
                x = local_states_tmp.col(i);
                sub2ind_nd(fsp_size, x, &adj_data.states_gid[i]);

                adj_data.num_edges[i] = 0;
                // Find indices of states connected to x on its row
                rx = arma::repmat(x, 1, stoichiometry.n_cols) - stoichiometry;
                irx = sub2ind_nd(fsp_size, rx);
                irx = arma::unique(irx);
                adj_data.rows_edge_ptr[i] = (int) adj_data.num_reachable_states_rows;
                sub2ind_nd(fsp_size, x, &adj_data.reachable_states_rows_gid[adj_data.num_reachable_states_rows]);
                adj_data.num_reachable_states_rows++;
                adj_data.num_edges[i]++;
                for (PetscInt j = 0; j < stoichiometry.n_cols; ++j) {
                    if (irx.at(j) > 0) {
                        adj_data.reachable_states_rows_gid[adj_data.num_reachable_states_rows] = (ZOLTAN_ID_TYPE) irx.at(
                                j);
                        adj_data.num_reachable_states_rows++;
                        adj_data.num_edges[i]++;
                    }
                }
                // Find indices of states connected to x on its column
                rx = arma::repmat(x, 1, stoichiometry.n_cols) + stoichiometry;
                irx = sub2ind_nd(fsp_size, rx);
                irx = arma::unique(irx);
                adj_data.cols_edge_ptr[i] = (int) adj_data.num_reachable_states_cols;
                for (PetscInt j = 0; j < stoichiometry.n_cols; ++j) {
                    if (irx.at(j) > 0) {
                        adj_data.reachable_states_cols_gid[adj_data.num_reachable_states_cols] = (ZOLTAN_ID_TYPE) irx.at(
                                j);
                        adj_data.num_reachable_states_cols++;
                        adj_data.num_edges[i]++;
                    }
                }
            }
            adj_data.rows_edge_ptr[n_local_tmp] = adj_data.num_reachable_states_rows;
            adj_data.cols_edge_ptr[n_local_tmp] = adj_data.num_reachable_states_cols;

            // Figure out which processors own the neighbors, this requires communication via Petsc's AO
            const PetscInt *procs_start;
            PetscInt num_procs;
            AO lex2petsc_tmp;
            arma::Row<PetscInt> local_idxs = sub2ind_nd(fsp_size, local_states_tmp);
            arma::Row<PetscInt> reachable_states_idxs;

            ierr = AOCreateMemoryScalable(comm, n_local_tmp, &local_idxs[0], NULL, &lex2petsc_tmp);
            CHKERRABORT(comm, ierr);

            MPI_Comm_rank(comm, &num_procs);
            ierr = PetscLayoutCreate(comm, &vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(vec_layout, n_local_tmp);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRanges(vec_layout, &procs_start);
            CHKERRABORT(comm, ierr);


            reachable_states_idxs.set_size(adj_data.num_reachable_states_rows);
            for (auto i{0}; i < adj_data.num_reachable_states_rows; ++i) {
                reachable_states_idxs(i) = (PetscInt) adj_data.reachable_states_rows_gid[i];
            }
            ierr = AOApplicationToPetsc(lex2petsc_tmp, adj_data.num_reachable_states_rows, &reachable_states_idxs[0]);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < adj_data.num_reachable_states_rows; ++i) {
                for (auto j{0}; j < num_procs; ++j) {
                    if ((procs_start[j] <= reachable_states_idxs[i]) &&
                        (procs_start[j + 1] > reachable_states_idxs[i])) {
                        adj_data.reachable_states_rows_proc[i] = j;
                    }
                }
            }

            reachable_states_idxs.set_size(adj_data.num_reachable_states_cols);
            for (auto i{0}; i < adj_data.num_reachable_states_cols; ++i) {
                reachable_states_idxs(i) = (PetscInt) adj_data.reachable_states_cols_gid[i];
            }
            ierr = AOApplicationToPetsc(lex2petsc_tmp, adj_data.num_reachable_states_cols, &reachable_states_idxs[0]);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < adj_data.num_reachable_states_cols; ++i) {
                for (auto j{0}; j < num_procs; ++j) {
                    if ((procs_start[j] <= reachable_states_idxs[i]) &&
                        (procs_start[j + 1] > reachable_states_idxs[i])) {
                        adj_data.reachable_states_cols_proc[i] = j;
                    }
                }
            }

            AODestroy(&lex2petsc_tmp);
            PetscLayoutDestroy(&vec_layout);
        }

        void FiniteStateSubsetGraph::free_graph_data() {
            Zoltan_Free((void **) &adj_data.reachable_states_rows_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.reachable_states_cols_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            delete[] adj_data.num_edges;
            delete[] adj_data.rows_edge_ptr;
            delete[] adj_data.cols_edge_ptr;
            delete[] adj_data.reachable_states_rows_proc;
            delete[] adj_data.reachable_states_cols_proc;
        }
    }
}
