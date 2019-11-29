//
// Created by Huy Vo on 6/2/19.
//

#include "FspMatrixConstrained.h"

namespace pacmensl {
FspMatrixConstrained::FspMatrixConstrained(MPI_Comm comm) : FspMatrixBase(comm)
{
}

/**
 * @brief Compute y = A*x.
 * @param t time.
 * @param x input vector.
 * @param y output vector.
 * @return error code, 0 if successful.
 */
int FspMatrixConstrained::Action(PetscReal t, Vec x, Vec y)
{
  int ierr;

  ierr = FspMatrixBase::Action(t, x, y); PACMENSLCHKERRQ(ierr);
  ierr = VecGetLocalVector(x, xx); CHKERRQ(ierr);
  ierr = VecSet(sink_entries_, 0.0); CHKERRQ(ierr);
  for (int i : tv_reactions_)
  {
    ierr = MatMult(tv_sinks_mat_[i], xx, sink_tmp); CHKERRQ(ierr);
    ierr = VecAXPY(sink_entries_, time_coefficients_[i], sink_tmp); CHKERRQ(ierr);
  }
  if (ti_sinks_mat_ != nullptr){
    ierr = MatMult(ti_sinks_mat_, xx, sink_tmp); CHKERRQ(ierr);
    ierr = VecAXPY(sink_entries_, 1.0, sink_tmp); CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(sink_scatter_ctx_, sink_entries_, y, ADD_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(sink_scatter_ctx_, sink_entries_, y, ADD_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecRestoreLocalVector(x, xx); CHKERRQ(ierr);
  return 0;
}

int FspMatrixConstrained::Destroy()
{
  PetscErrorCode ierr;
  FspMatrixBase::Destroy();
  if (sink_entries_ != nullptr)
  {
    ierr = VecDestroy(&sink_entries_); CHKERRQ(ierr);
  }
  if (sink_tmp != nullptr)
  {
    ierr = VecDestroy(&sink_tmp); CHKERRQ(ierr);
  }
  if (!tv_sinks_mat_.empty())
  {
    for (int i{0}; i < tv_sinks_mat_.size(); ++i)
    {
      if (tv_sinks_mat_[i])
      {
        ierr = MatDestroy(&tv_sinks_mat_[i]); CHKERRQ(ierr);
      }
    }
    tv_sinks_mat_.clear();
  }
  if (xx != nullptr){
    ierr = VecDestroy(&xx); CHKERRQ(ierr);
  }
  if (ti_sinks_mat_){
    ierr = MatDestroy(&ti_sinks_mat_); CHKERRQ(ierr);
  }
  if (sink_scatter_ctx_ != nullptr)
  {
    ierr = VecScatterDestroy(&sink_scatter_ctx_); CHKERRQ(ierr);
  }
  return 0;
}

FspMatrixConstrained::~FspMatrixConstrained()
{
  Destroy();
}

/**
* @brief Generate the local data structure for the FSP-truncated CME matrix with multiple sink states. This routine is collective.
* @param state_set set of CME states included in the finite state projection. For this particular class of matrix, state_set must be an instance of StateSetConstrained class.
* @param SM stoichiometry matrix
* @param prop propensity function, passed as callable object with signature <int(const int, const int, const int, const int, const int*, double* , void* )>. See also: PropFun.
* @param prop_args pointer to additional data for propensity function.
* @param new_prop_t callable object for evaluating the time coefficients. See also TcoefFun.
* @param prop_t_args pointer to additional data for time function.
* @return error code, 0 if successful.
*/
PacmenslErrorCode FspMatrixConstrained::GenerateValues(const StateSetBase &state_set,
                                                       const arma::Mat<Int> &SM,
                                                       std::vector<int> time_vayring,
                                                       const TcoefFun &new_prop_t,
                                                       const PropFun &prop,
                                                       const std::vector<int> &enable_reactions,
                                                       void *prop_t_args,
                                                       void *prop_args)
{
  PetscErrorCode ierr{0};

  auto *constrained_fss_ptr = dynamic_cast<const StateSetConstrained *>(&state_set);
  if (!constrained_fss_ptr) ierr = -1; PACMENSLCHKERRQ(ierr);

  sinks_rank_ = comm_size_ - 1; // rank of the processor that holds sink states_
  try
  {
    num_constraints_ = constrained_fss_ptr->GetNumConstraints();
  } catch (std::runtime_error &err)
  {
    ierr = -1; PACMENSLCHKERRQ(ierr);
  }

  // Generate the entries corresponding to usual states_
  ierr = FspMatrixBase::GenerateValues(state_set,
                                       SM,
                                       time_vayring,
                                       new_prop_t,
                                       prop,
                                       enable_reactions,
                                       prop_t_args,
                                       prop_args); PACMENSLCHKERRQ(ierr);

  // Generate the extra blocks corresponding to sink states_
  const arma::Mat<int> &state_list    = constrained_fss_ptr->GetStatesRef();
  int                  n_local_states = constrained_fss_ptr->GetNumLocalStates();
  int                  n_constraints  = constrained_fss_ptr->GetNumConstraints();
  arma::Mat<int>       can_reach_my_state;

  arma::Mat<Int>                    reachable_from_X(state_list.n_rows, state_list.n_cols);
  // Workspace for checking constraints
  arma::Mat<PetscInt>               constraints_satisfied(n_local_states, n_constraints);
  arma::Col<PetscInt>               nconstraints_satisfied;
  arma::Mat<int>                    d_nnz(n_constraints, num_reactions_);
  std::vector<arma::Row<int>>       sink_inz(n_constraints * num_reactions_);
  std::vector<arma::Row<PetscReal>> sink_rows(n_constraints * num_reactions_);

  tv_sinks_mat_.resize(num_reactions_);
  for (auto i_reaction : enable_reactions_)
  {
    // Count nnz for rows that represent sink states_
    can_reach_my_state = state_list + arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    ierr               = constrained_fss_ptr->CheckConstraints(n_local_states, can_reach_my_state.colptr(0),
                                                               constraints_satisfied.colptr(0)); PACMENSLCHKERRQ(ierr);
    nconstraints_satisfied = arma::sum(constraints_satisfied, 1);

    for (int  i_constr = 0; i_constr < n_constraints; ++i_constr)
    {
      d_nnz(i_constr, i_reaction) = n_local_states - arma::sum(constraints_satisfied.col(i_constr));
      sink_inz.at(n_constraints * i_reaction + i_constr).set_size(d_nnz(i_constr, i_reaction));
      sink_rows.at(n_constraints * i_reaction + i_constr).set_size(d_nnz(i_constr, i_reaction));
    }
    // Store the column indices and values of the nonzero entries on the sink rows
    for (int  i_constr = 0; i_constr < n_constraints; ++i_constr)
    {
      int      count   = 0;
      for (int i_state = 0; i_state < n_local_states; ++i_state)
      {
        if (constraints_satisfied(i_state, i_constr) == 0)
        {
          sink_inz.at(n_constraints * i_reaction + i_constr).at(count) = i_state;
          prop(i_reaction, state_list.n_rows, 1, state_list.colptr(i_state),
               &sink_rows.at(n_constraints * i_reaction + i_constr)[count], prop_args);
          count += 1;
        }
      }
    }
  }

  // Fill values for the time varying matrix
  for (auto i_reaction: tv_reactions_){
    ierr = MatCreate(PETSC_COMM_SELF, &tv_sinks_mat_[i_reaction]); CHKERRQ(ierr);
    ierr = MatSetType(tv_sinks_mat_[i_reaction], MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(tv_sinks_mat_[i_reaction], n_constraints, num_rows_local_, n_constraints, num_rows_local_); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(tv_sinks_mat_[i_reaction], n_constraints * n_local_states, NULL); CHKERRQ(ierr);
    for (auto i_constr{0}; i_constr < n_constraints; i_constr++)
    {
      ierr = MatSetValues(tv_sinks_mat_[i_reaction], 1, &i_constr, d_nnz(i_constr, i_reaction),
                          sink_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                          sink_rows.at(i_reaction * n_constraints + i_constr).memptr(), ADD_VALUES); CHKERRQ(ierr);
    }
    ierr               = MatAssemblyBegin(tv_sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tv_sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Fill values for the time-invariant matrix
  if (~ti_reactions_.empty()){
    ierr = MatCreate(PETSC_COMM_SELF, &ti_sinks_mat_); CHKERRQ(ierr);
    ierr = MatSetType(ti_sinks_mat_, MATSELL); CHKERRQ(ierr);
    ierr = MatSetSizes(ti_sinks_mat_, n_constraints, num_rows_local_, n_constraints, num_rows_local_); CHKERRQ(ierr);
    ierr = MatSeqSELLSetPreallocation(ti_sinks_mat_, n_constraints * n_local_states, NULL); CHKERRQ(ierr);
    for (auto i_reaction: ti_reactions_){
      for (auto i_constr{0}; i_constr < n_constraints; i_constr++){
        ierr = MatSetValues(ti_sinks_mat_, 1, &i_constr, d_nnz(i_constr, i_reaction),
                            sink_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                            sink_rows.at(i_reaction * n_constraints + i_constr).memptr(), ADD_VALUES); CHKERRQ(ierr);
      }
    }
    ierr               = MatAssemblyBegin(ti_sinks_mat_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ti_sinks_mat_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Local vectors for computing sink entries
  ierr = VecCreateSeq(PETSC_COMM_SELF, n_constraints, &sink_entries_); CHKERRQ(ierr);
  ierr = VecSetUp(sink_entries_); CHKERRQ(ierr);

  ierr = VecDuplicate(sink_entries_, &sink_tmp); CHKERRQ(ierr);
  ierr = VecSetUp(sink_tmp); CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &xx); CHKERRQ(ierr);
  ierr = VecSetUp(xx); CHKERRQ(ierr);

  // Scatter context for adding sink values
  int      *sink_global_indices = new int[n_constraints];
  for (int i{0}; i < n_constraints; ++i)
  {
    sink_global_indices[i] = constrained_fss_ptr->GetNumGlobalStates() + i;
  }
  IS       sink_is;
  ierr = ISCreateGeneral(comm_, n_constraints, sink_global_indices, PETSC_COPY_VALUES, &sink_is); CHKERRQ(ierr);
  ierr = VecScatterCreate(sink_entries_, NULL, work_, sink_is, &sink_scatter_ctx_); CHKERRQ(ierr);
  ierr = ISDestroy(&sink_is); CHKERRQ(ierr);
  delete[] sink_global_indices;
  return 0;
}

/**
 * @brief
 * @param fsp
 * @return
 */
PacmenslErrorCode FspMatrixConstrained::DetermineLayout_(const StateSetBase &fsp)
{
  PetscErrorCode ierr;

  num_rows_local_ = fsp.GetNumLocalStates();
  if (rank_ == sinks_rank_) num_rows_local_ += num_constraints_;

  // Generate matrix layout from Fsp's layout
  ierr = VecCreate(comm_, work_.mem()); CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_); CHKERRQ(ierr);
  ierr = VecSetSizes(work_, num_rows_local_, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(work_); CHKERRQ(ierr);
  return 0;
}

}