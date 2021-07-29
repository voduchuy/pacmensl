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

#include "StateSetBase.h"
#include "StateSetConstrained.h"

pacmensl::StateSetConstrained::StateSetConstrained(MPI_Comm new_comm)
    : StateSetBase(new_comm) {
}


PetscInt pacmensl::StateSetConstrained::CheckValidityStates(PetscInt num_states, PetscInt *x, PetscInt *out)
{
  int ierr;
  int *fval;
  fval = new int[rhs_constr.n_elem];
  for (int istate{0}; istate < num_states; ++istate){
    out[istate] = 0;
    for (int i1{0}; i1 < num_species_; ++i1) {
      if (x[istate*num_species_ + i1] < 0) {
        out[istate] = -1;
      }
    }
    ierr = lhs_constr(num_species_, rhs_constr.n_elem, 1, x + num_species_*istate, fval, args_constr);
    PACMENSLCHKERRQ(ierr);

    for (int i{0}; i < rhs_constr.n_elem; ++i) {
      if (fval[i] > rhs_constr(i)) {
        out[istate] = -1;
      }
    }
  }
  delete[] fval;
  return 0;
}

/**
 * Call level: local.
 * @param num_states : number of states. x : array of states. satisfied: output array of size num_states*num_constraints.
 * @return void.
 */
int
pacmensl::StateSetConstrained::CheckConstraints(PetscInt num_states, PetscInt *x, PetscInt *satisfied) const {
  auto *fval = new int[num_states * rhs_constr.n_elem];
  int ierr = lhs_constr(num_species_, rhs_constr.n_elem, num_states, x, fval, args_constr);
  PACMENSLCHKERRQ(ierr);

  for (int iconstr = 0; iconstr < rhs_constr.n_elem; ++iconstr) {
    for (int i = 0; i < num_states; ++i) {
      satisfied[num_states * iconstr + i] = (fval[rhs_constr.n_elem * i + iconstr] <=
          rhs_constr(iconstr)) ? 1 : 0;
      for (int j = 0; j < num_species_; ++j) {
        if (x[num_species_ * i + j] < 0) {
          satisfied[num_states * iconstr + i] = 1;
        }
      }
    }
  }
  delete[] fval;
  return 0;
}

arma::Row<int> pacmensl::StateSetConstrained::GetShapeBounds() const {
  return arma::Row<int>(rhs_constr);
}

PetscInt pacmensl::StateSetConstrained::GetNumConstraints() const {
  return rhs_constr.n_elem;
}

int
pacmensl::StateSetConstrained::default_constr_fun(int num_species, int num_constr, int n_states, int *states,
                                                  int *outputs, void *args) {
  for (int i{0}; i < n_states * num_species; ++i) {
    outputs[i] = states[i];
  }
  return 0;
}

int pacmensl::StateSetConstrained::SetShape(const fsp_constr_multi_fn &lhs_fun, arma::Row<int> &rhs_bounds,
                                            void *args) {
  lhs_constr = lhs_fun;
  rhs_constr = rhs_bounds;
  args_constr = args;
  return 0;
}

PacmenslErrorCode
pacmensl::StateSetConstrained::SetShape(int num_constraints, const fsp_constr_multi_fn &lhs_fun, int *bounds,
                                        void *args) {
  lhs_constr = lhs_fun;
  rhs_constr = arma::Row<int>(bounds, num_constraints);
  args_constr = args;
  return 0;
}

PacmenslErrorCode pacmensl::StateSetConstrained::SetShapeBounds(arma::Row<int> &rhs_bounds) {
  rhs_constr = rhs_bounds;
  return 0;
}

PacmenslErrorCode pacmensl::StateSetConstrained::SetShapeBounds(int num_constraints, int *bounds) {
  rhs_constr = arma::Row<int>(bounds, num_constraints);
  return 0;
}

/**
 * Call level: collective.
 * This function also distribute the states into the processors to improve the load-balance of matrix-vector multplications.
 */
PacmenslErrorCode pacmensl::StateSetConstrained::Expand() {
  int ierr;
  bool frontier_empty;

  // Switch states_ with status -1 to 1, for they may Expand to new states_ when the shape constraints are relaxed
  retrieve_state_status();
  arma::uvec iupdate = find(local_states_status_ == -1);
  arma::Mat<int> states_update;
  arma::Row<char> new_status;
  if (iupdate.n_elem > 0) {
    new_status.set_size(iupdate.n_elem);
    new_status.fill(1);
    states_update = local_states_.cols(iupdate);
  } else {
    states_update.set_size(num_species_, 0);
    new_status.set_size(0);
  }
  update_state_status(states_update, new_status);

  retrieve_state_status();
  frontier_lids_ = find(local_states_status_ == 1);

  logger_.event_begin(logger_.state_exploration_event);
  // Check if the set of frontier states_ are empty on all processors
  {
    int n1, n2;
    n1 = ( int ) frontier_lids_.n_elem;
    ierr = MPI_Allreduce(&n1, &n2, 1, MPI_INT, MPI_MAX, comm_);
    CHKERRMPI(ierr);
    frontier_empty = (n2 == 0);
  }

  arma::Row<char> frontier_status;
  while (!frontier_empty) {
    // Distribute frontier states_ to all processors
    frontiers_ = local_states_.cols(frontier_lids_);
    distribute_frontiers();
    frontier_status.set_size(frontiers_.n_cols);
    frontier_status.fill(0);

    arma::Mat<PetscInt> Y(num_species_, frontiers_.n_cols * num_reactions_);
    arma::Row<PetscInt> ystatus(Y.n_cols);

    for (int i{0}; i < frontiers_.n_cols; i++) {
      for (int j{0}; j < num_reactions_; ++j) {
        Y.col(j * frontiers_.n_cols + i) = frontiers_.col(i) + stoichiometry_matrix_.col(j);
      }
    }

    ierr = CheckValidityStates(Y.n_cols, Y.colptr(0), &ystatus[0]);
    PACMENSLCHKERRQ(ierr);

    for (int i{0}; i < frontiers_.n_cols; i++) {
      for (int j{0}; j < num_reactions_; ++j) {
        if (ystatus(j * frontiers_.n_cols + i) < 0) {
          frontier_status(i) = -1;
        }
      }
    }

    Y = Y.cols(find(ystatus == 0));
    Y = unique_columns(Y);
    ierr = AddStates(Y);
    PACMENSLCHKERRQ(ierr);

    // Deactivate states_ whose neighbors have all been explored and added to the state set
    update_state_status(frontiers_, frontier_status);
    retrieve_state_status();
    frontier_lids_ = find(local_states_status_ == 1);

    // Check if the set of frontier states_ are empty on all processors
    {
      int n1, n2;
      n1 = ( int ) frontier_lids_.n_elem;
      ierr = MPI_Allreduce(&n1, &n2, 1, MPI_INT, MPI_MAX, comm_); CHKERRMPI(ierr);
      frontier_empty = (n2 == 0);
    }
  }
  logger_.event_end(logger_.state_exploration_event);

  logger_.event_begin(logger_.call_partitioner_event);
  if (comm_size_ > 1) {
    if (num_global_states_old_ * (1.0 + lb_threshold_) <= 1.0 * num_global_states_ || num_global_states_old_ == 0) {
      num_global_states_old_ = num_global_states_;
      load_balance();
    }
  }
  logger_.event_end(logger_.call_partitioner_event);
  return 0;
}

PacmenslErrorCode pacmensl::StateSetConstrained::SetUp() {
  PacmenslErrorCode ierr;
  ierr = StateSetBase::SetUp();
  PACMENSLCHKERRQ(ierr);
  if (lhs_constr == nullptr){
    if (num_species_ != rhs_constr.n_elem) {
      PetscPrintf(comm_, "The number of constraint bounds when using default constraint must equal the number of species.\n");
      PACMENSLCHKERRQ(-1);
    }
    lhs_constr = default_constr_fun;
  }
  return 0;
}


