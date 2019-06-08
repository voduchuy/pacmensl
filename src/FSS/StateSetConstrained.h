//
// Created by Huy Vo on 5/31/19.
//

#ifndef PFSPAT_STATESUBSETCONSTRAINED_H
#define PFSPAT_STATESUBSETCONSTRAINED_H

#include <petscis.h>
#include "StateSetBase.h"

namespace cme {
    namespace parallel {
        class StateSetConstrained : public StateSetBase {
        protected:

            /// Left and right hand side for the custom constraints
            fsp_constr_multi_fn *lhs_constr;
            arma::Row<int> rhs_constr;

            inline int check_state(PetscInt *x);

            static void default_constr_fun(int num_species, int num_constr, int n_states, int *states, int *outputs);

        public:
            StateSetConstrained(MPI_Comm new_comm, int num_species, PartitioningType lb_type = Graph,
                                PartitioningApproach lb_approach = Repartition);

            void CheckConstraints(PetscInt num_states, PetscInt *x, PetscInt *satisfied);

            arma::Row<int> GetShapeBounds() const;

            int GetNumConstraints() const;

            void SetShape(fsp_constr_multi_fn *lhs_fun,
                          arma::Row<int> &rhs_bounds);

            void SetShapeBounds(arma::Row<PetscInt> &rhs_bounds);

            void Expand() override;
        };
    }
}

#endif //PFSPAT_STATESUBSETCONSTRAINED_H
