//
// Created by Huy Vo on 2019-06-24.
//

#include "StationaryFspMatrixConstrained.h"

pacmensl::StationaryFspMatrixConstrained::StationaryFspMatrixConstrained(MPI_Comm comm) : FspMatrixBase(comm)
{

}

PacmenslErrorCode pacmensl::StationaryFspMatrixConstrained::GenerateValues(const StateSetBase &fsp,
                                                                           const arma::Mat<Int> &SM,
                                                                           std::vector<int> time_vayring,
                                                                           const TcoefFun &new_t_fun,
                                                                           const PropFun &prop,
                                                                           const std::vector<int> &enable_reactions,
                                                                           void *t_fun_args,
                                                                           void *prop_args)
{
  int ierr{0};
  ierr = FspMatrixBase::GenerateValues(fsp,
                                       SM, std::vector<int>(),
                                       new_t_fun,
                                       prop,
                                       enable_reactions,
                                       t_fun_args,
                                       prop_args); PACMENSLCHKERRQ(ierr);

  // Generate the sink matrices
  auto *constrained_fss_ptr = dynamic_cast<const StateSetConstrained *>(&fsp);
  if (!constrained_fss_ptr) ierr = -1; PACMENSLCHKERRQ(ierr);

  try
  {
    num_constraints_ = constrained_fss_ptr->GetNumConstraints();
  } catch (std::runtime_error &err)
  {
    ierr = -1; PACMENSLCHKERRQ(ierr);
  }

  // Generate the extra blocks corresponding to sink states_
  const arma::Mat<int>              &state_list    = constrained_fss_ptr->GetStatesRef();
  int                               n_local_states = constrained_fss_ptr->GetNumLocalStates();
  int                               n_constraints  = constrained_fss_ptr->GetNumConstraints();
  arma::Mat<int>                    can_reach_my_state;
  arma::Mat<Int>                    reachable_from_X(state_list.n_rows, state_list.n_cols);
  // Workspace for checking constraints
  arma::Mat<PetscInt>               constraints_satisfied(n_local_states, n_constraints);
  arma::Col<PetscInt>               nconstraints_satisfied;
  arma::Mat<int>                    d_nnz(n_constraints, num_reactions_);
  std::vector<arma::Row<int>>       sink_inz(n_constraints * num_reactions_);
  std::vector<arma::Row<PetscReal>> sink_rows(n_constraints * num_reactions_);

  sinks_mat_.resize(num_reactions_);
  for (int i_reaction : enable_reactions_)
  {
    ierr = MatCreate(PETSC_COMM_SELF, sinks_mat_[i_reaction].mem()); CHKERRQ(ierr);
    ierr = MatSetType(sinks_mat_[i_reaction], MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(sinks_mat_[i_reaction], n_constraints, num_rows_local_, n_constraints, num_rows_local_); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(sinks_mat_[i_reaction], n_constraints * n_local_states, NULL); CHKERRQ(ierr);
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
          ierr = prop(i_reaction, state_list.n_rows, 1, state_list.colptr(i_state),
                      &sink_rows.at(n_constraints * i_reaction + i_constr)[count], prop_args); PACMENSLCHKERRQ(ierr);
          count += 1;
        }
      }
    }
    for (auto i_constr{0}; i_constr < n_constraints; i_constr++)
    {
      ierr = MatSetValues(sinks_mat_[i_reaction], 1, &i_constr, d_nnz(i_constr, i_reaction),
                          sink_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                          sink_rows.at(i_reaction * n_constraints + i_constr).memptr(), ADD_VALUES); CHKERRQ(ierr);
    }
    ierr               = MatAssemblyBegin(sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Local vectors for computing sink entries
  ierr = VecCreateSeq(PETSC_COMM_SELF, n_constraints, &sink_entries_); CHKERRQ(ierr);
  ierr = VecSetUp(sink_entries_); CHKERRQ(ierr);

  ierr = VecDuplicate(sink_entries_, &sink_tmp); CHKERRQ(ierr);
  ierr = VecSetUp(sink_tmp); CHKERRQ(ierr);

  // Generate the diagonal
  ierr = new_t_fun(0.0, num_reactions_, &time_coefficients_[0], t_fun_args_); PACMENSLCHKERRQ(ierr);
  ierr = VecCreate(comm_, &diagonal_); CHKERRQ(ierr);
  ierr = VecSetSizes(diagonal_, GetNumLocalRows(), PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(diagonal_); CHKERRQ(ierr);
  ierr = VecSet(diagonal_, 0.0); CHKERRQ(ierr);
  PetscReal *d_array;
  ierr = VecGetArray(diagonal_, &d_array); CHKERRQ(ierr);
  int                    num_states = fsp.GetNumLocalStates();
  std::vector<PetscReal> d_vals(num_states);
  for (auto i : enable_reactions_)
  {
    ierr = prop(i, state_list.n_rows, num_states, &state_list[0], &d_vals[0], prop_args); PACMENSLCHKERRQ(ierr);
    for (int j = 0; j < num_states; ++j)
    {
      d_array[j] += (-1.0)*time_coefficients_(i) * d_vals[j];
    }
  }
  ierr = VecRestoreArray(diagonal_, &d_array); CHKERRQ(ierr);
  return ierr;
}

int pacmensl::StationaryFspMatrixConstrained::Destroy()
{
  int ierr;
  sinks_mat_.clear();
  ierr = VecDestroy(&diagonal_); CHKERRQ(ierr);
  ierr = VecDestroy(&sink_entries_); CHKERRQ(ierr);
  ierr = VecDestroy(&sink_tmp); CHKERRQ(ierr);
  ierr = FspMatrixBase::Destroy(); PACMENSLCHKERRQ(ierr);
  return 0;
}

pacmensl::StationaryFspMatrixConstrained::~StationaryFspMatrixConstrained()
{
  Destroy();
}

int pacmensl::StationaryFspMatrixConstrained::Action(PetscReal t, Vec x, Vec y)
{
  int ierr;
  // Compute the 'usual' part of the matmult operation
  ierr = FspMatrixBase::Action(t, x, y); PACMENSLCHKERRQ(ierr);
  // Compute the sinks and direct them to the designated state
  ierr = VecGetLocalVector(x, xx); CHKERRQ(ierr);
  ierr = VecSet(sink_entries_, 0.0); CHKERRQ(ierr);
  for (int i : enable_reactions_)
  {
    ierr = MatMult(sinks_mat_[i], xx, sink_tmp); CHKERRQ(ierr);
    ierr = VecAXPY(sink_entries_, time_coefficients_[i], sink_tmp); CHKERRQ(ierr);
  }
  ierr       = VecRestoreLocalVector(x, xx); CHKERRQ(ierr);
  PetscReal sink_sum;
  ierr = VecSum(sink_entries_, &sink_sum); CHKERRQ(ierr);
  PetscReal sink_total;
  PetscReal *ylocal;
  ierr = VecGetArray(y, &ylocal); CHKERRQ(ierr);
  ierr = MPI_Reduce(( void * ) &sink_sum, ( void * ) &sink_total, 1, MPIU_REAL, MPIU_SUM, 0, comm_); CHKERRMPI(ierr);
  if (rank_ == 0){
    ylocal[0] += sink_total;
  }
  ierr = VecRestoreArray(y, &ylocal); CHKERRQ(ierr);
  return 0;
}

int pacmensl::StationaryFspMatrixConstrained::EvaluateOutflows(Vec sfsp_solution, arma::Row<PetscReal> &sinks)
{
  int ierr;
  ierr = t_fun_(0.0, num_reactions_, time_coefficients_.memptr(), t_fun_args_); PACMENSLCHKERRQ(ierr);
  ierr = VecGetLocalVector(sfsp_solution, xx); CHKERRQ(ierr);
  ierr = VecSet(sink_entries_, 0.0); CHKERRQ(ierr);
  for (int i : enable_reactions_)
  {
    ierr = MatMult(sinks_mat_[i], xx, sink_tmp); CHKERRQ(ierr);
    ierr = VecAXPY(sink_entries_, time_coefficients_[i], sink_tmp); CHKERRQ(ierr);
  }
  ierr       = VecRestoreLocalVector(sfsp_solution, xx); CHKERRQ(ierr);
  sinks.resize(num_constraints_);
  const PetscReal *local_sinks;
  ierr = VecGetArrayRead(sink_entries_, &local_sinks); CHKERRQ(ierr);
  ierr = MPI_Allreduce(( void * ) local_sinks, ( void * ) sinks.memptr(), num_constraints_, MPIU_REAL, MPIU_SUM, comm_); CHKERRQ(ierr);
  return 0;
}
