//
// Created by Huy Vo on 5/31/19.
//

#ifndef PACMENSL_STATESUBSETCONSTRAINED_H
#define PACMENSL_STATESUBSETCONSTRAINED_H

#include <petscis.h>
#include "StateSetBase.h"

namespace pacmensl {
typedef std::function<int(int, int, int, int *, int *,
                          void *)> fsp_constr_multi_fn;

class StateSetConstrained : public StateSetBase {
 public:
  StateSetConstrained(MPI_Comm new_comm, int num_species, PartitioningType lb_type = GRAPH,
                      PartitioningApproach lb_approach = REPARTITION);

  int CheckConstraints(PetscInt num_states, PetscInt *x, PetscInt *satisfied) const;

  arma::Row<int> GetShapeBounds() const;

  int GetNumConstraints() const;

  void SetShape(const fsp_constr_multi_fn &lhs_fun, arma::Row<int> &rhs_bounds, void *args = nullptr);

  void SetShape(int num_constraints, const fsp_constr_multi_fn &lhs_fun, int *bounds, void *args = nullptr);

  void SetShapeBounds(arma::Row<PetscInt> &rhs_bounds);

  void SetShapeBounds(int num_constraints, int *bounds);

  void Expand() override;

 protected:

  /// Left and right hand side for the custom constraints
  fsp_constr_multi_fn lhs_constr = nullptr;
  arma::Row<int> rhs_constr;
  void *args_constr = nullptr;

  inline int check_state(PetscInt *x);

  static int
  default_constr_fun(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args);
};
}

#endif //PACMENSL_STATESUBSETCONSTRAINED_H
