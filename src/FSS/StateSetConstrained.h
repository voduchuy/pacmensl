//
// Created by Huy Vo on 5/31/19.
//

#ifndef PARALLEL_FSP_STATESUBSETCONSTRAINED_H
#define PARALLEL_FSP_STATESUBSETCONSTRAINED_H

#include <petscis.h>
#include "StateSetBase.h"

namespace cme{
    namespace parallel{
        class StateSetConstrained : public StateSetBase {
        protected:

            /// Left and right hand side for the custom constraints
            fsp_constr_multi_fn *lhs_constr;
            arma::Row< int > rhs_constr;

            inline int check_state( PetscInt *x );

            static void default_constr_fun( int num_species, int num_constr, int n_states, int *states, int *outputs );

        public:
            StateSetConstrained(MPI_Comm new_comm, int num_species, PartitioningType lb_type = Graph, PartitioningApproach lb_approach = Repartition);

            void check_constraint_on_proc( PetscInt num_states, PetscInt *x, PetscInt *satisfied );
            arma::Row< int > get_shape_bounds( );

            int get_num_constraints( );
            void set_shape( fsp_constr_multi_fn *lhs_fun,
                            arma::Row< int > &rhs_bounds );
            void set_shape_bounds( arma::Row< PetscInt > &rhs_bounds );

            void expand( ) override;
        };
    }
}

#endif //PARALLEL_FSP_STATESUBSETCONSTRAINED_H
