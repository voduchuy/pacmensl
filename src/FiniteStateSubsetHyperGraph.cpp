//
// Created by Huy Vo on 12/15/18.
//

#include <FiniteStateSubsetHyperGraph.h>

#include "FiniteStateSubsetHyperGraph.h"


namespace cme {
    namespace parallel {
        FiniteStateSubsetHyperGraph::FiniteStateSubsetHyperGraph(MPI_Comm new_comm) : FiniteStateSubset(new_comm) {
            partitioning_type = HyperGraph;
            Zoltan_Set_Param(zoltan, "LB_METHOD", "HYPERGRAPH");
            Zoltan_Set_Param(zoltan, "HYPERGRAPH_PACKAGE", "PHG");
            Zoltan_Set_Param(zoltan, "RETURN_LISTS", "PARTS");
            Zoltan_Set_Param(zoltan, "DEBUG_LEVEL", "0");
            Zoltan_Set_Param(zoltan, "ADD_OBJ_WEIGHT", "PINS");
            Zoltan_Set_Param(zoltan, "PHG_RANDOMIZE_INPUT", "1");
            Zoltan_Set_Param(zoltan, "OBJ_WEIGHT_DIM", "0"); // use Zoltan default vertex weights
            Zoltan_Set_Param(zoltan, "EDGE_WEIGHT_DIM", "0");// use Zoltan default hyperedge weights

            PetscLogEventRegister("Generate Hypergraph data", 0, &generate_hg_event);
            PetscLogEventRegister("Call Zoltan_LB", 0, &call_zoltan_event);
            PetscLogEventRegister("Generate AO", 0, &generate_ao_event);
        }

        FiniteStateSubsetHyperGraph::~FiniteStateSubsetHyperGraph() {
            Zoltan_Destroy(&zoltan);
        }

        void FiniteStateSubsetHyperGraph::GenerateStatesAndOrdering() {
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0) {
                throw std::runtime_error(
                        "FiniteStateSubset: stoichiometry is required for HyperGraph partioning type.");
            }

            PetscErrorCode ierr;

            //
            // Set Zoltan mode to partitioning from scratch
            //
            Zoltan_Set_Param(zoltan, "LB_APPROACH", "PARTITION");

            //
            // Initial temporary partitioning based on lexicographic ordering
            //
            arma::Mat<PetscInt> local_states_tmp = get_my_naive_local_states();
            PetscInt n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            //
            // Create the hypergraph data
            //
            PetscLogEventBegin(generate_hg_event, 0, 0, 0, 0);
            generate_hypergraph_data(local_states_tmp);
            PetscLogEventEnd(generate_hg_event, 0, 0, 0, 0);

            //
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            //
            PetscLogEventBegin(call_zoltan_event, 0, 0, 0, 0);
            call_zoltan_partitioner();
            PetscLogEventEnd(call_zoltan_event, 0, 0, 0, 0);

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            PetscLogEventBegin(generate_ao_event, 0, 0, 0, 0);
            compute_petsc_ordering_from_zoltan();
            PetscLogEventEnd(generate_ao_event, 0, 0, 0, 0);

            //
            // Generate local states
            //
            get_local_states_from_ao();

            free_hypergraph_data();
            free_zoltan_part_variables();
        }

        void FiniteStateSubsetHyperGraph::ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) {
            PetscErrorCode ierr;
            assert(new_fsp_size.n_elem == fsp_size.n_elem);
            for (auto i{0}; i < fsp_size.n_elem; ++i) {
                assert(new_fsp_size(i) >= fsp_size(i));
            }
            if (local_states.n_elem == 0) {
                PetscPrintf(comm,
                            "FiniteStateSubsetHyperGraph: found empty local states array, probably because GenerateStatesAndOrdering was never called.\n");
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
            // Create the hypergraph data
            //
            PetscLogEventBegin(generate_hg_event, 0, 0, 0, 0);
            generate_hypergraph_data(local_states_tmp);
            PetscLogEventEnd(generate_hg_event, 0, 0, 0, 0);

            //
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            //
            PetscLogEventBegin(call_zoltan_event, 0, 0, 0, 0);
            call_zoltan_partitioner();
            PetscLogEventEnd(call_zoltan_event, 0, 0, 0, 0);

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            PetscLogEventBegin(generate_ao_event, 0, 0, 0, 0);
            compute_petsc_ordering_from_zoltan();
            PetscLogEventEnd(generate_ao_event, 0, 0, 0, 0);

            //
            // Generate local states
            //
            get_local_states_from_ao();

            free_hypergraph_data();
            free_zoltan_part_variables();
        }

        void FiniteStateSubsetHyperGraph::generate_hypergraph_data(arma::Mat<PetscInt> &local_states) {
            PetscInt n_local_tmp = local_states.n_cols;
            PetscInt nnz, i_here;
            arma::Col<PetscInt> x((size_t) n_species);
            arma::Mat<PetscInt> brx((size_t) n_species, (size_t) stoichiometry.n_cols);
            arma::Row<PetscInt> irx((size_t) stoichiometry.n_cols);
            ZOLTAN_ID_PTR vtx_gid;
            int *vtx_edge_ptr;
            ZOLTAN_ID_PTR pin_gid;

            vtx_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(n_local_tmp * sizeof(ZOLTAN_ID_TYPE), __FILE__, __LINE__);
            vtx_edge_ptr = new int[n_local_tmp];
            pin_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc((1 + stoichiometry.n_cols) * n_local_tmp * sizeof(ZOLTAN_ID_TYPE),
                                                    __FILE__, __LINE__);

            nnz = 0;
            for (PetscInt i = 0; i < n_local_tmp; ++i) {
                x = local_states.col(i);
                // Find indices of states connected to x
                brx = arma::repmat(x, 1, stoichiometry.n_cols) - stoichiometry;
                irx = sub2ind_nd(fsp_size, brx);
                irx = arma::unique(irx);

                sub2ind_nd(fsp_size, x, &vtx_gid[i]);
                vtx_edge_ptr[i] = (int) nnz;
                pin_gid[nnz] = vtx_gid[i];
                nnz++;
                for (PetscInt j = 0; j < stoichiometry.n_cols; ++j) {
                    if (irx.at(j) > 0) {
                        pin_gid[nnz] = (ZOLTAN_ID_TYPE) irx.at(j);
                        nnz++;
                    }
                }
            }
            adj_data.num_local_states = n_local_tmp;
            adj_data.num_reachable_states_rows = nnz;
            adj_data.rows_edge_ptr = vtx_edge_ptr;
            adj_data.reachable_states_rows_gid = pin_gid;
            adj_data.states_gid = vtx_gid;
        }

        void FiniteStateSubsetHyperGraph::free_hypergraph_data() {
            Zoltan_Free((void **) &adj_data.states_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &adj_data.reachable_states_rows_gid, __FILE__, __LINE__);
            delete[] adj_data.rows_edge_ptr;
        }
    }
}