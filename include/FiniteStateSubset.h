//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSET_H
#define PARALLEL_FSP_FINITESTATESUBSET_H

#include "cme_util.h"

namespace cme {
    namespace petsc {
        enum PartioningType {
            Linear, ParMetis, NotSet
        };

        class FiniteStateSubset {
        protected:
            int set_up = 0;
            int stoich_set = 0;
            MPI_Comm comm;

            PartioningType partitioning_type;
            arma::Row<PetscInt> fsp_size;
            PetscInt n_species;
            PetscInt n_states_global;
            PetscInt n_local_states;
            arma::Mat<PetscInt> local_states;
            arma::Mat<PetscInt> stoichiometry;

            PetscLayout vec_layout = nullptr;
            AO lex2petsc = nullptr;
        public:

            // Generic Interface
            explicit FiniteStateSubset(MPI_Comm new_comm);

            MPI_Comm GetComm();

            arma::Row<PetscInt> GetFSPSize();

            void SetStoichiometry(arma::Mat<PetscInt> SM);

            void SetSize(arma::Row<PetscInt> &new_fsp_size);

            PetscInt GetNumLocalStates();

            PetscInt GetNumSpecies();

            AO GetAO();

            void PrintAO();

            arma::Mat<PetscInt> GetLocalStates();

            std::tuple<PetscInt, PetscInt> GetLayoutStartEnd();

            arma::Row<PetscInt> State2Petsc(arma::Mat<PetscInt> state);

            arma::Row<PetscReal> SinkStatesReduce(Vec P);

            void Destroy();

            ~FiniteStateSubset();

            // Implementation-dependent methods
            // This procedure generate data for the members:
            // local_states, vec_layout, lex2petsc
            virtual void GenerateStatesAndOrdering() {};

            friend arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);
        };

        arma::Col<PetscReal> marginal(FiniteStateSubset &fsp, Vec P, PetscInt species);

    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSET_H
