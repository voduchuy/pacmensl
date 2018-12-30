//
// Created by Huy Vo on 12/4/18.
//

#include <FSS/FiniteStateSubsetRCB.h>

namespace cme {
    namespace parallel {
        FiniteStateSubsetRCB::FiniteStateSubsetRCB(MPI_Comm new_comm) : FiniteStateSubset(new_comm) {
            partitioning_type = RCB;
            Zoltan_Set_Param(zoltan, "LB_METHOD", "RCB");
            Zoltan_Set_Param(zoltan, "RETURN_LISTS", "PARTS");
            Zoltan_Set_Param(zoltan, "DEBUG_LEVEL", "0");
            Zoltan_Set_Param(zoltan, "OBJ_WEIGHT_DIM", "0"); // use Zoltan default vertex weights
        }

        void FiniteStateSubsetRCB::GenerateStatesAndOrdering() {
            PetscErrorCode ierr;
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0) {
                throw std::runtime_error("FiniteStateSubset: stoichiometry is required for Graph partitioning type.");
            }

            Zoltan_Set_Param(zoltan, "LB_APPROACH", "PARTITION");

            //
            // Initial temporary partitioning based on lexicographic ordering
            //
            arma::Mat<PetscInt> local_states_tmp = compute_my_naive_local_states();

            //
            // Generate Geometrical data
            //
            GenerateGeomData(fsp_size, local_states_tmp);
            GenerateVertexData(local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with Petsc's IS
            //
            CallZoltanLoadBalancing();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            ComputePetscOrderingFromZoltan();

            // Generate local states
            LocalStatesFromAO();

            FreeVertexData();
            FreeGeomData();
            FreeZoltanParts();
        }

        void FiniteStateSubsetRCB::ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) {
            PetscErrorCode ierr;
            assert(new_fsp_size.n_elem == fsp_size.n_elem);
            for (auto i{0}; i < fsp_size.n_elem; ++i) {
                assert(new_fsp_size(i) >= fsp_size(i));
            }
            if (local_states.n_cols == 0) {
                PetscPrintf(PETSC_COMM_SELF,
                            "FiniteStateSubsetGraph: found empty local states array, probably because GenerateStatesAndOrdering was never called.\n");
                MPI_Abort(comm, -1);
            }
            arma::Row<PetscInt> fsp_size_old = fsp_size;
            SetSize(new_fsp_size);

            if (lex2petsc) AODestroy(&lex2petsc);
            if (vec_layout) PetscLayoutDestroy(&vec_layout);

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
            // Create the geometrical data
            //
            GenerateGeomData(new_fsp_size, local_states_tmp);
            GenerateVertexData(local_states_tmp);

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

            FreeVertexData();
            FreeGeomData();
            FreeZoltanParts();
        }

    }
}
