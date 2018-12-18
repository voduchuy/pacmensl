//
// Created by Huy Vo on 12/4/18.
//


#include "FiniteStateSubsetNaive.h"

namespace cme {
    namespace parallel {
        void FiniteStateSubsetNaive::GenerateStatesAndOrdering() {
            PetscInt ierr;
            PetscLayout layout_without_sinks;
            PetscInt local_size_without_sinks, global_size_with_sinks;
            PetscInt start1, end1, start2, end2;

            PetscMPIInt rank;
            MPI_Comm_rank(comm, &rank);

            ierr = PetscLayoutCreate(comm, &layout_without_sinks);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetSize(layout_without_sinks, n_states_global);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(layout_without_sinks, PETSC_DECIDE);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(layout_without_sinks);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetLocalSize(layout_without_sinks, &local_size_without_sinks);
            CHKERRABORT(comm, ierr);

            ierr = PetscLayoutCreate(comm, &vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(vec_layout, local_size_without_sinks + ((PetscInt) fsp_size.n_elem));
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetSize(vec_layout, &global_size_with_sinks);
            CHKERRABORT(comm, ierr);

            ierr = PetscLayoutGetRange(layout_without_sinks, &start1, &end1);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRange(vec_layout, &start2, &end2);
            CHKERRABORT(comm, ierr);

            // Count number of local states
            n_local_states = local_size_without_sinks;

            arma::Row<PetscInt> fsp_indices(local_size_without_sinks + n_species);
            arma::Row<PetscInt> petsc_indices(local_size_without_sinks + n_species);
            // Add values for local indices for usual states
            for (PetscInt i = 0; i < local_size_without_sinks; ++i){
                fsp_indices.at(i) = start1 + i;
                petsc_indices.at(i) = start2 + i;
            }
            // Add indices for sink states
            for (PetscInt i = 0; i < n_species; ++i){
                fsp_indices[local_size_without_sinks + i] = n_states_global + rank*n_species + i;
                petsc_indices[local_size_without_sinks + n_species -1 -i] = end2 -1 -i;
            }

            // Create the AO object that maps from lexicographic ordering to Petsc Vec ordering
            IS fsp_is, petsc_is;
            ISCreateGeneral(comm, local_size_without_sinks + n_species, &fsp_indices[0], PETSC_COPY_VALUES, &fsp_is);
            ISCreateGeneral(comm, local_size_without_sinks + n_species, &petsc_indices[0], PETSC_COPY_VALUES, &petsc_is);
            AOCreate(comm, &lex2petsc);
            AOSetIS(lex2petsc, fsp_is, petsc_is);
            AOSetType(lex2petsc, AOMEMORYSCALABLE);
            AOSetFromOptions(lex2petsc);
            CHKERRABORT(comm, ISDestroy(&fsp_is));
            CHKERRABORT(comm, ISDestroy(&petsc_is));

//            ierr = AOCreateBasic(comm, local_size_without_sinks + n_species, &fsp_indices[0], &petsc_indices[0], &lex2petsc);

            // Generate the local states
            arma::Row<PetscInt> petsc_local_indices(n_local_states);
            CHKERRABORT(comm, PetscLayoutGetRange(vec_layout, &petsc_local_indices[0], &petsc_local_indices[n_local_states-1]));
            for (PetscInt i{0}; i < n_local_states; ++i){
                petsc_local_indices[i] = petsc_local_indices[0] + i;
            }
            CHKERRABORT(comm, AOPetscToApplication(lex2petsc, n_local_states, &petsc_local_indices[0]));
            local_states = ind2sub_nd<arma::Mat<PetscInt>>(fsp_size, petsc_local_indices);

            ierr = PetscLayoutDestroy(&layout_without_sinks);
            CHKERRABORT(comm, ierr);
            set_up = 1;
        }
    }
}
