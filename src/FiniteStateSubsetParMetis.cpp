//
// Created by Huy Vo on 12/4/18.
//

#include <FiniteStateSubsetParMetis.h>

#include "FiniteStateSubsetParMetis.h"

namespace cme{
    namespace petsc{
        void FiniteStateSubsetParMetis::GenerateStatesAndOrdering() {
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0){
                throw std::runtime_error("FintieStateSubset: stoichiometry is required for ParMetis partioning type.");
            }

            // Create the adjacency matrix
            Mat adj;
            PetscInt local_start, local_end;

            CHKERRABORT(comm, MatCreate(comm, &adj));
            CHKERRABORT(comm, MatSetSizes(adj, PETSC_DECIDE, PETSC_DECIDE, n_states_global, n_states_global));
            CHKERRABORT(comm, MatSetFromOptions(adj));
            CHKERRABORT(comm, MatMPIAIJSetPreallocation(adj,  2*stoichiometry.n_cols + 1, NULL, 2*stoichiometry.n_cols + 1, NULL));
            CHKERRABORT(comm, MatSetUp(adj));
            CHKERRABORT(comm, MatGetOwnershipRange(adj, &local_start, &local_end));
            arma::Row<PetscReal> onevec(2*stoichiometry.n_cols);
            onevec.fill(1.0);
            arma::Col<PetscInt> x((size_t) n_species);
            for (PetscInt i{local_start}; i < local_end; ++i)
            {
                PetscInt i_here = i;
                x = ind2sub_nd(fsp_size, i_here);

                // Find indices of states connected to x
                arma::Mat<PetscInt> rx = arma::repmat(x, 1, stoichiometry.n_cols) + stoichiometry;
                arma::Mat<PetscInt> brx = arma::repmat(x, 1, stoichiometry.n_cols) - stoichiometry;
                rx = arma::join_horiz(rx, brx);
                arma::Row<PetscInt> irx = sub2ind_nd(fsp_size, rx);

                MatSetValues(adj, 1, &i, 1, &i, &onevec[0], INSERT_VALUES);
                MatSetValues(adj, 1, &i, (PetscInt) irx.n_elem, &irx[0], &onevec[0], INSERT_VALUES);
            }
            MatAssemblyBegin(adj, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(adj, MAT_FINAL_ASSEMBLY);

            // Create partitioning and generate new global indices for the states
            MatPartitioning partitioning;
            IS processor_id, global_numbering;
            CHKERRABORT(comm, ISCreate(comm, &processor_id));
            CHKERRABORT(comm, ISCreate(comm, &global_numbering));
            CHKERRABORT(comm, MatPartitioningCreate(comm, &partitioning));
            CHKERRABORT(comm, MatPartitioningSetAdjacency(partitioning, adj));
            CHKERRABORT(comm, MatPartitioningSetType(partitioning, MATPARTITIONINGPARMETIS));
            CHKERRABORT(comm, MatPartitioningSetFromOptions(partitioning));
            CHKERRABORT(comm, MatPartitioningApply(partitioning, &processor_id));
            CHKERRABORT(comm, ISPartitioningToNumbering(processor_id, &global_numbering));

            const PetscInt *proc_id_array, *global_numbering_array;
            CHKERRABORT(comm, ISGetIndices(processor_id, &proc_id_array));
            CHKERRABORT(comm, ISGetIndices(global_numbering, &global_numbering_array));
            // Figure out how many states each processor own by reducing the proc_id_array
            PetscMPIInt my_rank, comm_size;
            MPI_Comm_size(comm, &comm_size);
            MPI_Comm_rank(comm, &my_rank);
            arma::Row<PetscInt> n_own(comm_size);
            ISPartitioningCount(processor_id, comm_size, &n_own[0]);
            n_local_states = n_own[my_rank];

            // Generate layout
            CHKERRABORT(comm, PetscLayoutCreate(comm, &vec_layout));
            CHKERRABORT(comm, PetscLayoutSetLocalSize(vec_layout, n_local_states + n_species));
            CHKERRABORT(comm, PetscLayoutSetSize(vec_layout, PETSC_DECIDE));
            CHKERRABORT(comm, PetscLayoutSetUp(vec_layout));

            PetscInt layout_start, layout_end;
            CHKERRABORT(comm, PetscLayoutGetRange(vec_layout, &layout_start, &layout_end));

            // Adjust global numbering to add sink states
            std::vector<PetscInt> corrected_global_numbering(local_end - local_start + n_species);
            for (PetscInt i{0}; i < local_end-local_start; ++i){
                corrected_global_numbering.at(i) = global_numbering_array[i] + proc_id_array[i]*((PetscInt) fsp_size.n_elem);
            }

            arma::Row<PetscInt> fsp_indices(local_end - local_start + n_species);
            for (PetscInt i = 0; i < local_end - local_start; ++i){
                fsp_indices.at(i) = local_start + i;
            }

            for (PetscInt i = 0; i < n_species; ++i){
                corrected_global_numbering[local_end - local_start + n_species - i -1 ] = layout_end - 1 -i;
                fsp_indices[local_end - local_start + n_species - i -1 ] = n_states_global + my_rank*n_species + i;
            }

            // Create the AO object that maps from lexicographic ordering to Petsc Vec ordering
            IS fsp_is, petsc_is;
            ISCreateGeneral(comm, local_end - local_start + n_species, &fsp_indices[0], PETSC_COPY_VALUES, &fsp_is);
            ISCreateGeneral(comm, local_end - local_start + n_species, &corrected_global_numbering[0], PETSC_COPY_VALUES, &petsc_is);
            AOCreate(comm, &lex2petsc);
            AOSetIS(lex2petsc, fsp_is, petsc_is);
            AOSetType(lex2petsc, AOMEMORYSCALABLE);
            AOSetFromOptions(lex2petsc);
            CHKERRABORT(comm, ISDestroy(&fsp_is));
            CHKERRABORT(comm, ISDestroy(&petsc_is));

//            AOCreateBasic(comm, local_end - local_start + n_species, &fsp_indices[0], &corrected_global_numbering[0], &lex2petsc);
            // Generate local states
            arma::Row<PetscInt> petsc_local_indices(n_local_states);
            CHKERRABORT(comm, PetscLayoutGetRange(vec_layout, &petsc_local_indices[0], &petsc_local_indices[n_local_states-1]));
            for (PetscInt i{0}; i < n_local_states; ++i){
                petsc_local_indices[i] = petsc_local_indices[0] + i;
            }
            CHKERRABORT(comm, AOPetscToApplication(lex2petsc, n_local_states, &petsc_local_indices[0]));
            local_states = ind2sub_nd(fsp_size, petsc_local_indices);

            CHKERRABORT(comm, ISRestoreIndices(processor_id, &proc_id_array));
            CHKERRABORT(comm, ISRestoreIndices(global_numbering, &global_numbering_array));
            CHKERRABORT(comm, ISDestroy(&processor_id));
            CHKERRABORT(comm, ISDestroy(&global_numbering));
            CHKERRABORT(comm, MatPartitioningDestroy(&partitioning));
            CHKERRABORT(comm, MatDestroy(&adj));
        }
    }
}
