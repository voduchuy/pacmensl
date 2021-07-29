/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef PACMENSL_STATESUBSETCONSTRAINED_H
#define PACMENSL_STATESUBSETCONSTRAINED_H

#include <petscis.h>
#include "StateSetBase.h"

namespace pacmensl {
typedef std::function<int(int, int, int, int *, int *,
                          void *)> fsp_constr_multi_fn;

class StateSetConstrained : public StateSetBase {
 public:
  explicit StateSetConstrained(MPI_Comm new_comm = MPI_COMM_WORLD);

  int CheckConstraints(PetscInt num_states, PetscInt *x, PetscInt *satisfied) const;

  arma::Row<int> GetShapeBounds() const;

  int GetNumConstraints() const;

  PacmenslErrorCode SetShape(const fsp_constr_multi_fn &lhs_fun, arma::Row<int> &rhs_bounds, void *args = nullptr);

  PacmenslErrorCode SetShape(int num_constraints, const fsp_constr_multi_fn &lhs_fun, int *bounds, void *args = nullptr);

  PacmenslErrorCode SetShapeBounds(arma::Row<PetscInt> &rhs_bounds);

  PacmenslErrorCode SetShapeBounds(int num_constraints, int *bounds);

  PacmenslErrorCode SetUp() override;

  PacmenslErrorCode Expand() override;

 protected:

  /// Left and right hand side for the custom constraints
  fsp_constr_multi_fn lhs_constr = nullptr;
  arma::Row<int> rhs_constr;
  void *args_constr = nullptr;

  inline PetscInt CheckValidityStates(PetscInt num_states, PetscInt *x, PetscInt *out);

  static int
  default_constr_fun(int num_species, int num_constr, int n_states, int *states, int *outputs, void *args);
};
}

#endif //PACMENSL_STATESUBSETCONSTRAINED_H
