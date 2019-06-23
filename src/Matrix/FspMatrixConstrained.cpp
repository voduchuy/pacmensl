//
// Created by Huy Vo on 6/2/19.
//

#include "FspMatrixConstrained.h"

namespace pacmensl {
FspMatrixConstrained::FspMatrixConstrained(MPI_Comm comm) : FspMatrixBase(comm) {
}

int FspMatrixConstrained::action(PetscReal t, Vec x, Vec y) {
  int ierr;
  FspMatrixBase::action(t, x, y);
  ierr = VecGetLocalVector(x, xx);
  CHKERRQ(ierr);
  ierr = VecSet(sink_entries_, 0.0);
  CHKERRQ(ierr);
  for (int i = 0; i < num_reactions_; ++i) {
    ierr = MatMult(sinks_mat_[i], xx, sink_tmp);
    CHKERRQ(ierr);
    ierr = VecAXPY(sink_entries_, time_coefficients_[i], sink_tmp);
    CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(sink_scatter_ctx_, sink_entries_, y, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(sink_scatter_ctx_, sink_entries_, y, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecRestoreLocalVector(x, xx);
  CHKERRQ(ierr);
  return 0;
}

int FspMatrixConstrained::Destroy() {
  PetscErrorCode ierr;
  FspMatrixBase::Destroy();
  if (sink_entries_ != nullptr) {
    ierr = VecDestroy(&sink_entries_);
    CHKERRQ(ierr);
  }
  if (sink_tmp != nullptr){
    ierr = VecDestroy(&sink_tmp);
    CHKERRQ(ierr);
  }
  if (!sinks_mat_.empty()) {
    for (int i{0}; i < sinks_mat_.size(); ++i)
    {
      ierr = MatDestroy(&sinks_mat_[i]);
      CHKERRQ(ierr);
    }
    sinks_mat_.clear();
  }
  if (sink_scatter_ctx_ != nullptr) {
    ierr = VecScatterDestroy(&sink_scatter_ctx_);
    CHKERRQ(ierr);
  }
  return 0;
}

FspMatrixConstrained::~FspMatrixConstrained() {
  Destroy();
}

int FspMatrixConstrained::GenerateValues(const StateSetBase &fsp, const arma::Mat<int> &SM, const PropFun &prop,
                                         void *prop_args, const TcoefFun &new_t_fun, void *t_fun_args) {
  PetscErrorCode ierr{0};

  const StateSetConstrained *constrained_fss_ptr = dynamic_cast<const StateSetConstrained *>(&fsp);
  if (!constrained_fss_ptr) ierr = -1;
  PACMENSLCHKERRQ(ierr);

  sinks_rank_ = comm_size_ - 1; // rank of the processor that holds sink states_
  try {
    num_constraints_ = constrained_fss_ptr->GetNumConstraints();
  } catch (std::runtime_error &err) {
    ierr = -1;
    PACMENSLCHKERRQ(ierr);
  }

  // Generate the entries corresponding to usual states_
  ierr = FspMatrixBase::GenerateValues(fsp, SM, prop, prop_args, new_t_fun, t_fun_args);
  PACMENSLCHKERRQ(ierr);

  // Generate the extra blocks corresponding to sink states_
  const arma::Mat<int> &state_list = constrained_fss_ptr->GetStatesRef();
  int n_local_states = constrained_fss_ptr->GetNumLocalStates();
  int n_constraints = constrained_fss_ptr->GetNumConstraints();
  arma::Mat<int> can_reach_my_state;

  arma::Mat<Int> reachable_from_X(state_list.n_rows, state_list.n_cols);
  // Workspace for checking constraints
  arma::Mat<PetscInt> constraints_satisfied(n_local_states, n_constraints);
  arma::Col<PetscInt> nconstraints_satisfied;
  arma::Mat<int> d_nnz(n_constraints, num_reactions_);
  std::vector<arma::Row<int>> sink_inz(n_constraints * num_reactions_);
  std::vector<arma::Row<PetscReal>> sink_rows(n_constraints * num_reactions_);

  sinks_mat_.resize(num_reactions_);
  for (int i_reaction{0}; i_reaction < num_reactions_; ++i_reaction) {
    ierr = MatCreate(PETSC_COMM_SELF, &sinks_mat_[i_reaction]);
    CHKERRQ(ierr);
    ierr = MatSetType(sinks_mat_[i_reaction], MATSEQAIJ);
    CHKERRQ(ierr);
    ierr = MatSetSizes(sinks_mat_[i_reaction], n_constraints, n_rows_local_, n_constraints, n_rows_local_);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(sinks_mat_[i_reaction], n_constraints * n_local_states, NULL);
    CHKERRQ(ierr);
    // Count nnz for rows that represent sink states_
    can_reach_my_state = state_list + arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    ierr = constrained_fss_ptr->CheckConstraints(n_local_states, can_reach_my_state.colptr(0),
                                                 constraints_satisfied.colptr(0));
    PACMENSLCHKERRQ(ierr);
    nconstraints_satisfied = arma::sum(constraints_satisfied, 1);

    for (int i_constr = 0; i_constr < n_constraints; ++i_constr) {
      d_nnz(i_constr, i_reaction) = n_local_states - arma::sum(constraints_satisfied.col(i_constr));
      sink_inz.at(n_constraints * i_reaction + i_constr).set_size(d_nnz(i_constr, i_reaction));
      sink_rows.at(n_constraints * i_reaction + i_constr).set_size(d_nnz(i_constr, i_reaction));
    }
    // Store the column indices and values of the nonzero entries on the sink rows
    for (int i_constr = 0; i_constr < n_constraints; ++i_constr) {
      int count = 0;
      for (int i_state = 0; i_state < n_local_states; ++i_state) {
        if (constraints_satisfied(i_state, i_constr) == 0) {
          sink_inz.at(n_constraints * i_reaction + i_constr).at(count) = i_state;
          prop(i_reaction, state_list.n_rows, 1, state_list.colptr(i_state),
               &sink_rows.at(n_constraints * i_reaction + i_constr)[count], prop_args);
          count += 1;
        }
      }
    }
    for (auto i_constr{0}; i_constr < n_constraints; i_constr++) {
      ierr = MatSetValues(sinks_mat_[i_reaction], 1, &i_constr, d_nnz(i_constr, i_reaction),
                          sink_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                          sink_rows.at(i_reaction * n_constraints + i_constr).memptr(), ADD_VALUES);
      CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }

  // Local vectors for computing sink entries
  ierr = VecCreateSeq(PETSC_COMM_SELF, n_constraints, &sink_entries_);
  CHKERRQ(ierr);
  ierr = VecSetUp(sink_entries_);
  CHKERRQ(ierr);

  ierr = VecDuplicate(sink_entries_, &sink_tmp);
  CHKERRQ(ierr);
  ierr = VecSetUp(sink_tmp);
  CHKERRQ(ierr);

  // Scatter context for adding sink values
  int *sink_global_indices = new int[n_constraints];
  for (int i{0}; i < n_constraints; ++i) {
    sink_global_indices[i] = constrained_fss_ptr->GetNumGlobalStates() + i;
  }
  IS sink_is;
  Vec tmp;
  ierr = ISCreateGeneral(comm_, n_constraints, sink_global_indices, PETSC_COPY_VALUES, &sink_is);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(sink_entries_, NULL, work_, sink_is, &sink_scatter_ctx_);
  CHKERRQ(ierr);
  ierr = ISDestroy(&sink_is);
  CHKERRQ(ierr);
  delete[] sink_global_indices;
}

int FspMatrixConstrained::DetermineLayout_(const StateSetBase &fsp) {
  PetscErrorCode ierr;

  n_rows_local_ = fsp.GetNumLocalStates();
  if (my_rank_ == sinks_rank_) n_rows_local_ += num_constraints_;

  // Generate matrix layout from FSP's layout
  ierr = VecCreate(comm_, &work_);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_);
  CHKERRQ(ierr);
  ierr = VecSetSizes(work_, n_rows_local_, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(work_);
  CHKERRQ(ierr);
}
}