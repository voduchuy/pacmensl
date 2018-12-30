//
// Created by Huy Vo on 12/15/18.
//

#include <FSS/FiniteStateSubsetHyperGraph.h>

#include "FiniteStateSubsetHyperGraph.h"


namespace cme {
    namespace parallel {
        FiniteStateSubsetHyperGraph::FiniteStateSubsetHyperGraph(MPI_Comm new_comm) : FiniteStateSubset(new_comm) {
            partitioning_type = HyperGraph;
            Zoltan_Set_Param(zoltan, "LB_METHOD", "HYPERGRAPH");
            Zoltan_Set_Param(zoltan, "HYPERGRAPH_PACKAGE", "PHG");
            Zoltan_Set_Param(zoltan, "CHECK_HYPERGRAPH", "0");
            Zoltan_Set_Param(zoltan, "RETURN_LISTS", "PARTS");
            Zoltan_Set_Param(zoltan, "DEBUG_LEVEL", "4");
            Zoltan_Set_Param(zoltan, "PHG_REPART_MULTIPLIER", "1000");
            Zoltan_Set_Param(zoltan, "PHG_RANDOMIZE_INPUT", "1");
            Zoltan_Set_Param(zoltan, "OBJ_WEIGHT_DIM", "1");
            Zoltan_Set_Param(zoltan, "EDGE_WEIGHT_DIM", "0");// use Zoltan default hyperedge weights
        }

        void FiniteStateSubsetHyperGraph::GenerateStatesAndOrdering() {
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0) {
                throw std::runtime_error(
                        "FiniteStateSubset: stoichiometry is required for HyperGraph partioning type.");
            }

            //
            // Set Zoltan mode to partitioning from scratch
            //
            Zoltan_Set_Param(zoltan, "LB_APPROACH", "PARTITION");

            //
            // Initial temporary partitioning based on lexicographic ordering
            //
            arma::Mat<PetscInt> local_states_tmp = compute_my_naive_local_states();
            PetscInt n_local_tmp = (PetscInt) local_states_tmp.n_cols;

            //
            // Create the hypergraph data
            //
            GenerateGeomData(fsp_size, local_states_tmp);
            GenerateHyperGraphData(local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            //
            CallZoltanLoadBalancing();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            ComputePetscOrderingFromZoltan();

            //
            // Generate local states
            //
            LocalStatesFromAO();

            FreeHyperGraphData();
            FreeGeomData();
            FreeZoltanParts();
        }

        void FiniteStateSubsetHyperGraph::ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) {
            PetscErrorCode ierr;
            assert(new_fsp_size.n_elem == fsp_size.n_elem);
            for (auto i{0}; i < fsp_size.n_elem; ++i) {
                assert(new_fsp_size(i) >= fsp_size(i));
            }
            if (n_local_states == 0) {
                PetscPrintf(PETSC_COMM_SELF,
                            "FiniteStateSubsetHyperGraph: found empty local states array, probably because GenerateStatesAndOrdering was never called.\n");
                MPI_Abort(comm, -1);
            }
            if (lex2petsc) AODestroy(&lex2petsc);
            if (vec_layout) PetscLayoutDestroy(&vec_layout);

            arma::Row<PetscInt> fsp_size_old = fsp_size;
            SetSize(new_fsp_size);

            //
            // Switch Zoltan to repartition
            //
            Zoltan_Set_Param(zoltan, "LB_APPROACH", zoltan_part_opt.c_str());

            //
            // Explore for new states that satisfy the new FSP bounds
            //

            PetscInt n_new_states;
            arma::Mat<PetscInt> new_candidates, local_states_tmp;
            arma::Row<PetscInt> is_new_states;

            new_candidates = compute_my_naive_local_states();
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
            GenerateGeomData(fsp_size, local_states_tmp);
            GenerateHyperGraphData(local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            //
            CallZoltanLoadBalancing();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            ComputePetscOrderingFromZoltan();

            //
            // Generate local states
            //
            LocalStatesFromAO();

            FreeHyperGraphData();
            FreeGeomData();
            FreeZoltanParts();
        }
    }
}