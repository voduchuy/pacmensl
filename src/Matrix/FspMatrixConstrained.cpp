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

#include "FspMatrixConstrained.h"

namespace pacmensl {
FspMatrixConstrained::FspMatrixConstrained(MPI_Comm comm) : FspMatrixBase(comm)
{
}


int FspMatrixConstrained::Action(PetscReal t,Vec x,Vec y)
{
  int ierr;

  ierr = FspMatrixBase::Action(t,x,y);
  PACMENSLCHKERRQ(ierr);
  ierr = VecGetLocalVectorRead(x,xx);
  CHKERRQ(ierr);
  ierr = VecSet(sink_entries_,0.0);
  CHKERRQ(ierr);
  if (!tv_reactions_.empty())
  {
    for (int i {0}; i < tv_reactions_.size(); ++i)
    {
      ierr = MatMult(tv_sinks_mat_[i],xx,sink_tmp);
      CHKERRQ(ierr);
      ierr = VecAXPY(sink_entries_,time_coefficients_[tv_reactions_[i]],sink_tmp);
      CHKERRQ(ierr);
    }
  }

  if (ti_sinks_mat_ != nullptr)
  {
    ierr = MatMult(ti_sinks_mat_,xx,sink_tmp);
    CHKERRQ(ierr);
    ierr = VecAXPY(sink_entries_,1.0,sink_tmp);
    CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(sink_scatter_ctx_,sink_entries_,y,ADD_VALUES,SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(sink_scatter_ctx_,sink_entries_,y,ADD_VALUES,SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecRestoreLocalVectorRead(x,xx);
  CHKERRQ(ierr);
  return 0;
}

int FspMatrixConstrained::Destroy()
{
  PetscErrorCode ierr;
  FspMatrixBase::Destroy();
  if (sink_entries_ != nullptr)
  {
    ierr = VecDestroy(&sink_entries_);
    CHKERRQ(ierr);
  }
  if (sink_tmp != nullptr)
  {
    ierr = VecDestroy(&sink_tmp);
    CHKERRQ(ierr);
  }
  if (!tv_sinks_mat_.empty())
  {
    for (int i{0}; i < tv_sinks_mat_.size(); ++i)
    {
      if (tv_sinks_mat_[i])
      {
        ierr = MatDestroy(&tv_sinks_mat_[i]);
        CHKERRQ(ierr);
      }
    }
    tv_sinks_mat_.clear();
  }
  if (xx != nullptr)
  {
    ierr = VecDestroy(&xx);
    CHKERRQ(ierr);
  }
  if (ti_sinks_mat_ != nullptr)
  {
    ierr = MatDestroy(&ti_sinks_mat_);
    CHKERRQ(ierr);
  }
  if (sink_scatter_ctx_ != nullptr)
  {
    ierr = VecScatterDestroy(&sink_scatter_ctx_);
    CHKERRQ(ierr);
  }
  sinkmat_nnz.clear();
  sinkmat_inz.clear();
  sinkmat_entries.clear();

  return 0;
}

FspMatrixConstrained::~FspMatrixConstrained()
{
  Destroy();
}


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

  PetscInt own_start,own_end;

  auto *constrained_fss_ptr = dynamic_cast<const StateSetConstrained *>(&state_set);
  if (!constrained_fss_ptr) ierr = -1;
  PACMENSLCHKERRQ(ierr);

  sinks_rank_ = comm_size_ - 1; // rank of the processor that holds sink states_
  try
  {
    num_constraints_ = constrained_fss_ptr->GetNumConstraints();
  } catch (std::runtime_error &err)
  {
    ierr = -1;
    PACMENSLCHKERRQ(ierr);
  }

  // Generate the entries corresponding to usual states_
  ierr = FspMatrixBase::GenerateValues(state_set,
                                       SM,
                                       time_vayring,
                                       new_prop_t,
                                       prop,
                                       enable_reactions,
                                       prop_t_args,
                                       prop_args);
  PACMENSLCHKERRQ(ierr);

  // Generate the extra blocks corresponding to sink states_
  const arma::Mat<int> &state_list    = constrained_fss_ptr->GetStatesRef();
  int                  n_local_states = constrained_fss_ptr->GetNumLocalStates();
  int                  n_constraints  = constrained_fss_ptr->GetNumConstraints();
  arma::Mat<int>       can_reach_my_state;

  arma::Mat<Int>      reachable_from_X(state_list.n_rows,state_list.n_cols);
  // Workspace for checking constraints
  arma::Mat<PetscInt> constraints_satisfied(n_local_states,n_constraints);

  sinkmat_nnz.resize(n_constraints,num_reactions_);
  sinkmat_inz.resize(n_constraints * num_reactions_);
  sinkmat_entries.resize(n_constraints * num_reactions_);
  tv_sinks_mat_.resize(tv_reactions_.size());
  for (auto i_reaction : enable_reactions_)
  {
    // Count nnz for rows that represent sink states_
    can_reach_my_state = state_list + arma::repmat(SM.col(i_reaction),1,state_list.n_cols);
    ierr               = constrained_fss_ptr->CheckConstraints(n_local_states,can_reach_my_state.colptr(0),
                                                               constraints_satisfied.colptr(0));
    PACMENSLCHKERRQ(ierr);

    for (int i_constr = 0; i_constr < n_constraints; ++i_constr)
    {
      sinkmat_nnz(i_constr,i_reaction) = n_local_states - arma::sum(constraints_satisfied.col(i_constr));
      sinkmat_inz.at(n_constraints * i_reaction + i_constr).set_size(sinkmat_nnz(i_constr,i_reaction));
      sinkmat_entries.at(n_constraints * i_reaction + i_constr).set_size(sinkmat_nnz(i_constr,i_reaction));
    }
    // Store the column indices and values of the nonzero entries on the sink rows
    for (int i_constr = 0; i_constr < n_constraints; ++i_constr)
    {
      int      count   = 0;
      for (int i_state = 0; i_state < n_local_states; ++i_state)
      {
        if (constraints_satisfied(i_state,i_constr) == 0)
        {
          sinkmat_inz.at(n_constraints * i_reaction + i_constr).at(count) = i_state;
          prop(i_reaction,state_list.n_rows,1,state_list.colptr(i_state),
               &sinkmat_entries.at(n_constraints * i_reaction + i_constr)[count],prop_args);
          count += 1;
        }
      }
    }
  }

  // Fill values for the time varying matrix
  for (int i{0}; i < tv_reactions_.size(); ++i)
  {
    int i_reaction = tv_reactions_[i];
    ierr = MatCreate(PETSC_COMM_SELF,&tv_sinks_mat_[i]);
    CHKERRQ(ierr);
    ierr = MatSetType(tv_sinks_mat_[i],MATSEQAIJ);
    CHKERRQ(ierr);
    ierr = MatSetSizes(tv_sinks_mat_[i],n_constraints,num_rows_local_,n_constraints,num_rows_local_);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(tv_sinks_mat_[i],NULL,sinkmat_nnz.colptr(i_reaction));
    CHKERRQ(ierr);
    for (auto i_constr{0}; i_constr < n_constraints; i_constr++)
    {
      ierr = MatSetValues(tv_sinks_mat_[i],1,&i_constr,sinkmat_nnz(i_constr,i_reaction),
                          sinkmat_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                          sinkmat_entries.at(i_reaction * n_constraints + i_constr).memptr(),ADD_VALUES);
      CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(tv_sinks_mat_[i],MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tv_sinks_mat_[i],MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }

  // Fill values for the time-invariant matrix
  if (!ti_reactions_.empty())
  {
    ierr = MatCreate(PETSC_COMM_SELF,&ti_sinks_mat_);
    CHKERRQ(ierr);
    ierr = MatSetType(ti_sinks_mat_,MATSELL);
    CHKERRQ(ierr);
    ierr = MatSetSizes(ti_sinks_mat_,n_constraints,num_rows_local_,n_constraints,num_rows_local_);
    CHKERRQ(ierr);
    ierr = MatSeqSELLSetPreallocation(ti_sinks_mat_,n_constraints * num_rows_local_,NULL);
    CHKERRQ(ierr);
    for (auto i_reaction: ti_reactions_)
    {
      for (auto i_constr{0}; i_constr < n_constraints; i_constr++)
      {
        ierr = MatSetValues(ti_sinks_mat_,1,&i_constr,sinkmat_nnz(i_constr,i_reaction),
                            sinkmat_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                            sinkmat_entries.at(i_reaction * n_constraints + i_constr).memptr(),ADD_VALUES);
        CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(ti_sinks_mat_,MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ti_sinks_mat_,MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }

  // Update sinkmat_inz to use global indices
  ierr = VecGetOwnershipRange(work_,&own_start,&own_end);
  CHKERRQ(ierr);
  for (auto i_reaction:enable_reactions_)
  {
    for (auto i_constr{0}; i_constr < n_constraints; i_constr++)
    {
      sinkmat_inz.at(i_reaction * n_constraints + i_constr) += own_start;
    }
  }

  // Local vectors for computing sink entries
  ierr = VecCreateSeq(PETSC_COMM_SELF,n_constraints,&sink_entries_);
  CHKERRQ(ierr);
  ierr = VecSetUp(sink_entries_);
  CHKERRQ(ierr);

  ierr = VecDuplicate(sink_entries_,&sink_tmp);
  CHKERRQ(ierr);
  ierr = VecSetUp(sink_tmp);
  CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,num_rows_local_,&xx);
  CHKERRQ(ierr);
  ierr = VecSetUp(xx);
  CHKERRQ(ierr);

  // Scatter context for adding sink values
  int      *sink_global_indices = new int[n_constraints];
  for (int i{0}; i < n_constraints; ++i)
  {
    sink_global_indices[i] = constrained_fss_ptr->GetNumGlobalStates() + i;
  }
  IS       sink_is;
  ierr = ISCreateGeneral(comm_,n_constraints,sink_global_indices,PETSC_COPY_VALUES,&sink_is);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(sink_entries_,NULL,work_,sink_is,&sink_scatter_ctx_);
  CHKERRQ(ierr);
  ierr = ISDestroy(&sink_is);
  CHKERRQ(ierr);
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
  ierr = VecCreate(comm_,work_.mem());
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_);
  CHKERRQ(ierr);
  ierr = VecSetSizes(work_,num_rows_local_,PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(work_);
  CHKERRQ(ierr);
  ierr = VecGetSize(work_,&num_rows_global_);
  CHKERRQ(ierr);
  return 0;
}

int FspMatrixConstrained::CreateRHSJacobian(Mat *A)
{
  int ierr;

  PetscInt       own_start,own_end,num_local_states,itmp;
  arma::Col<Int> d_nz,o_nz,tmp;

  ierr = MatCreate(comm_,A);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);
  CHKERRQ(ierr);
  ierr = MatSetSizes(*A,num_rows_local_,num_rows_local_,num_rows_global_,num_rows_global_);
  CHKERRQ(ierr);
  ierr = MatSetUp(*A);
  CHKERRQ(ierr);

  num_local_states = offdiag_vals_.n_rows;
  d_nz.set_size(num_rows_local_);
  o_nz.set_size(num_rows_local_);
  d_nz.fill(1);
  o_nz.fill(0);
  for (auto ir: tv_reactions_)
  {
    for (int i{0}; i < num_local_states; ++i)
    {
      d_nz(i) += (dblock_nz_(i,ir) - 1);
      o_nz(i) += oblock_nz_(i,ir);
    }
  }

  if (!ti_reactions_.empty())
  {
    for (int i{0}; i < num_local_states; ++i)
    {
      d_nz(i) += (ti_dblock_nz_(i) - 1);
      o_nz(i) += ti_oblock_nz_(i);
    }
  }


  // Find nnz for rows of the sink states
  tmp.set_size(num_constraints_);
  tmp.zeros();
  if (rank_ == sinks_rank_)
  {
    for (auto ir: enable_reactions_)
    {
      for (int i = 0; i < num_constraints_; ++i)
      {
        d_nz(num_local_states + i) += sinkmat_nnz(i,ir);
      }
    }
  } else
  {
    for (auto ir: enable_reactions_)
    {
      tmp += sinkmat_nnz.col(ir);
    }
  }

  ierr =
      MPI_Reduce(&tmp[0],o_nz.memptr() + num_local_states,num_constraints_,MPIU_INT,MPIU_SUM,sinks_rank_,comm_);
  CHKERRMPI(ierr);

  if (rank_ == sinks_rank_){
    for (int i = 0; i < num_constraints_; ++i)
    {
      d_nz(num_local_states + i) = std::min(d_nz(num_local_states+i), num_rows_local_);
      o_nz(num_local_states + i) = std::min(o_nz(num_local_states+i), num_rows_global_ - num_rows_local_);
    }
  }

  ierr = MatMPIAIJSetPreallocation(*A,PETSC_NULL,&d_nz[0],PETSC_NULL,&o_nz[0]);
  CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(work_,&own_start,&own_end);
  CHKERRQ(ierr);

  for (auto ir: enable_reactions_)
  {
    for (int i{0}; i < num_local_states; ++i)
    {
      ierr = MatSetValue(*A,own_start + i,offdiag_col_idxs_(i,ir),0.0,INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }
  for (int  i{0}; i < num_rows_local_; ++i)
  {
    ierr = MatSetValue(*A,own_start + i,own_start + i,0.0,INSERT_VALUES);
    CHKERRQ(ierr);
  }

  for (auto i_reaction: enable_reactions_)
  {
    for (int i_constr{0}; i_constr < num_constraints_; i_constr++)
    {
      for (int j{0}; j < sinkmat_nnz(i_constr,i_reaction); j++)
      {
        itmp = num_rows_global_ - num_constraints_ + i_constr;
        ierr = MatSetValue(*A,itmp,*(sinkmat_inz.at(i_reaction * num_constraints_ + i_constr).memptr() + j),
                           0.0,INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }

  MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);

  return 0;
}

int FspMatrixConstrained::ComputeRHSJacobian(PetscReal t,Mat A)
{
  int      ierr = 0;
  PetscInt itmp,jtmp;
  PetscInt own_start,own_end;

  ierr = FspMatrixBase::ComputeRHSJacobian(t,A);
  PACMENSLCHKERRQ(ierr);

  ierr = VecGetOwnershipRange(work_,&own_start,&own_end);
  CHKERRQ(ierr);

  PetscReal atmp;
  if (!tv_reactions_.empty())
  {
    for (int i_reaction: tv_reactions_)
    {
      for (int i_constr{0}; i_constr < num_constraints_; i_constr++)
      {

        for (int j{0}; j < sinkmat_nnz(i_constr,i_reaction); j++)
        {
          itmp = num_rows_global_ - num_constraints_ + i_constr;
          atmp = time_coefficients_(i_reaction) * sinkmat_entries.at(i_reaction * num_constraints_ + i_constr)(j);
          jtmp = sinkmat_inz.at(i_reaction * num_constraints_ + i_constr)(j);

          ierr = MatSetValue(A,itmp,jtmp,atmp,ADD_VALUES);
          CHKERRQ(ierr);

        }
      }
    }
  }

  if (!ti_reactions_.empty())
  {
    for (auto i_reaction: ti_reactions_)
    {
      for (auto i_constr{0}; i_constr < num_constraints_; i_constr++)
      {
        itmp = num_rows_global_ - num_constraints_ + i_constr;
        ierr = MatSetValues(A,1,&itmp,sinkmat_nnz(i_constr,i_reaction),
                            sinkmat_inz.at(i_reaction * num_constraints_ + i_constr).memptr(),
                            sinkmat_entries.at(i_reaction * num_constraints_ + i_constr).memptr(),
                            ADD_VALUES);
        CHKERRQ(ierr);
      }
    }
  }

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  return 0;
}

PacmenslErrorCode FspMatrixConstrained::GetLocalMVFlops(PetscInt *nflops)
{
  PetscInt ierr;
  MatInfo minfo;

  ierr = FspMatrixBase::GetLocalMVFlops(nflops); PACMENSLCHKERRQ(ierr);

  ierr = MatGetInfo(ti_sinks_mat_, MAT_LOCAL, &minfo); CHKERRQ(ierr);
  *nflops += 2*((PetscInt) minfo.nz_used);

  for (int j{0}; j < tv_sinks_mat_.size(); ++j){
    ierr = MatGetInfo(tv_sinks_mat_[j], MAT_LOCAL, &minfo); CHKERRQ(ierr);
    *nflops += 2*((PetscInt) minfo.nz_used) + num_constraints_;
  }

  return 0;
}

}