//
// Created by Huy Vo on 12/4/18.
//
#include <FiniteStateSubset.h>

#include "FiniteStateSubset.h"

namespace cme {
    namespace petsc {
        FiniteStateSubset::FiniteStateSubset(MPI_Comm new_comm) {
            int ierr;
            comm = new_comm;
            partitioning_type = NotSet;
            local_states.resize(0);
            fsp_size.resize(0);
            n_states_global = 0;
            stoichiometry.resize(0);
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

        arma::Mat<PetscInt> FiniteStateSubset::GetStates() {
            arma::Mat<PetscInt> states_return = local_states;
            return states_return;
        }

        arma::Row<PetscInt> FiniteStateSubset::State2Petsc(arma::Mat<PetscInt> state) {
            arma::Row<PetscInt> lex_indices = cme::sub2ind_nd(fsp_size, state);
            AOApplicationToPetsc(lex2petsc, (PetscInt) lex_indices.n_elem, &lex_indices[0]);
            return lex_indices;
        }

        FiniteStateSubset::~FiniteStateSubset() {
            int ierr;
            comm = nullptr;
            ierr = AODestroy(&lex2petsc);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutDestroy(&vec_layout);
            CHKERRABORT(comm, ierr);
        }

        std::tuple<PetscInt, PetscInt> FiniteStateSubset::GetLayoutStartEnd() {
            int start, end, ierr;
            ierr = PetscLayoutGetRange(vec_layout, &start, &end);
            CHKERRABORT(comm, ierr);
            return std::make_tuple(start, end);

        }

        arma::Row<PetscReal> FiniteStateSubset::SinkStatesReduce(Vec P) {
            int ierr;

            arma::Row<PetscReal> local_sinks(fsp_size.n_elem), global_sinks(fsp_size.n_elem);

            int p_local_size;
            ierr = VecGetLocalSize(P, &p_local_size); CHKERRABORT(comm, ierr);

            if (p_local_size != local_states.n_cols){
                printf("FiniteStateSubset::SinkStatesReduce: The layout of P and FiniteStateSubset do not match.\n");
                MPI_Abort(comm, 1);
            }

            PetscReal *p_data;
            VecGetArray(P, &p_data);
            for (int i{0}; i < fsp_size.n_elem; ++i){
                local_sinks(i) = p_data[p_local_size -1 - i];
                ierr = MPI_Allreduce(&local_sinks[i], &global_sinks[i], 1, MPIU_REAL, MPI_SUM, comm);
                CHKERRABORT(comm, ierr);
            }

            return global_sinks;
        }
    }
}
