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
            CHKERRABORT(comm, AOApplicationToPetsc(lex2petsc, (PetscInt) lex_indices.n_elem, &lex_indices[0]));
            return lex_indices;
        }

        void FiniteStateSubset::State2Petsc(arma::Mat<PetscInt> state, PetscInt *indx) {
            cme::sub2ind_nd(fsp_size, state, indx);
            CHKERRABORT(comm, AOApplicationToPetsc(lex2petsc, (PetscInt) state.n_cols, indx));
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
