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

#include "Sys.h"
#include "FspMatrixBase.h"

namespace pacmensl {

FspMatrixBase::FspMatrixBase(MPI_Comm comm) {
  comm_ = comm;
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &comm_size_);
}

int FspMatrixBase::Action(PetscReal t, Vec x, Vec y) {
  PetscInt ierr;
  
  ierr = VecSet(y, 0.0);
  CHKERRQ(ierr);
  if (has_values_ == PETSC_FALSE) return 0;

  if (!tv_reactions_.empty()) {
    ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
    PACMENSLCHKERRQ(ierr);
    
    for (int ir{0}; ir < tv_mats_.size(); ++ir) {
      ierr = MatMult(tv_mats_[ir], x, work_);
      CHKERRQ(ierr);
      ierr = VecAXPY(y, time_coefficients_[tv_reactions_[ir]], work_);
      CHKERRQ(ierr);
    }
  }
  
  if (*ti_mat_.mem() != nullptr) {
    ierr = MatMult(ti_mat_, x, work_);
    CHKERRQ(ierr);
    ierr = VecAXPY(y, 1.0, work_);
    CHKERRQ(ierr);
  }
  return 0;
}

PacmenslErrorCode FspMatrixBase::GenerateValues(const StateSetBase &fsp, const Model &model) {
  return GenerateValues(fsp,
                        model.stoichiometry_matrix_,
                        model.tv_reactions_,
                        model.prop_t_,
                        model.prop_x_,
                        std::vector<int>(),
                        model.prop_t_args_,
                        model.prop_x_args_
                        );
}

PacmenslErrorCode FspMatrixBase::GenerateValues(const StateSetBase &fsp,
                                                const arma::Mat<Int> &SM,
                                                std::vector<int> time_varying,
                                                const TcoefFun &new_prop_t,
                                                const PropFun &new_prop_x,
                                                const std::vector<int> &enable_reactions,
                                                void *prop_t_args,
                                                void *prop_x_args) {
  PacmenslErrorCode ierr;
  PetscInt n_species, n_local_states, own_start, own_end;
  const arma::Mat<Int> &state_list = fsp.GetStatesRef();
  arma::Mat<Int> can_reach_my_state(state_list.n_rows, state_list.n_cols);
  
  ierr = DetermineLayout_(fsp);
  PACMENSLCHKERRQ(ierr);
  
  // Get the global number of rows
  ierr = VecGetSize(work_, &num_rows_global_);
  CHKERRQ(ierr);
  
  n_species = fsp.GetNumSpecies();
  n_local_states = fsp.GetNumLocalStates();
  num_reactions_ = fsp.GetNumReactions();
  time_coefficients_.set_size(num_reactions_);
  
  t_fun_ = new_prop_t;
  t_fun_args_ = prop_t_args;
  
  enable_reactions_ = enable_reactions;
  if (enable_reactions_.empty()) {
    enable_reactions_ = std::vector<int>(num_reactions_);
    for (int i = 0; i < num_reactions_; ++i) {
      enable_reactions_[i] = i;
    }
  }
  for (int ir: enable_reactions_) {
    if (std::find(time_varying.begin(), time_varying.end(), ir) != time_varying.end()) {
      tv_reactions_.push_back(ir);
    } else {
      ti_reactions_.push_back(ir);
    }
  }
  tv_mats_.resize(tv_reactions_.size());
  
  // Find the nnz per row of diagonal and off-diagonal matrices
  offdiag_col_idxs_.set_size(n_local_states, num_reactions_);
  offdiag_vals_.set_size(n_local_states, num_reactions_);
  diag_vals_.set_size(n_local_states, num_reactions_);
  dblock_nz_.set_size(num_rows_local_, num_reactions_);
  oblock_nz_.set_size(num_rows_local_, num_reactions_);
  dblock_nz_.fill(1);
  oblock_nz_.zeros();
  
  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);
  // Count nnz for matrix rows
  for (auto i_reaction : enable_reactions_) {
    can_reach_my_state = state_list - arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    fsp.State2Index(can_reach_my_state, offdiag_col_idxs_.colptr(i_reaction));
    new_prop_x(i_reaction, can_reach_my_state.n_rows, can_reach_my_state.n_cols, &can_reach_my_state[0],
               offdiag_vals_.colptr(i_reaction), prop_x_args);
    
    for (auto i_state{0}; i_state < n_local_states; ++i_state) {
      if (offdiag_col_idxs_(i_state, i_reaction) >= own_start && offdiag_col_idxs_(i_state, i_reaction) < own_end) {
        dblock_nz_(i_state, i_reaction) += 1;
      } else if (offdiag_col_idxs_(i_state, i_reaction) >= 0) {
        oblock_nz_(i_state, i_reaction) += 1;
      }
    }
  }
  
  // Fill values for the time-varying part
  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);
  MatType mtype;
  for (int i{0}; i < tv_reactions_.size(); ++i) {
    int i_reaction = tv_reactions_[i];
    ierr = MatCreate(comm_, tv_mats_[i].mem());
    CHKERRQ(ierr);
    ierr = MatSetType(tv_mats_[i], MATMPISELL);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(tv_mats_[i]);
    CHKERRQ(ierr);
    ierr = MatSetSizes(tv_mats_[i], num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_);
    CHKERRQ(ierr);
    ierr = MatGetType(tv_mats_[i], &mtype);
    CHKERRQ(ierr);
    if ((strcmp(mtype, MATSELL) == 0) || (strcmp(mtype, MATMPISELL) == 0) || (strcmp(mtype, MATSEQSELL) == 0)) {
      ierr = MatMPISELLSetPreallocation(tv_mats_[i],
                                        PETSC_NULL,
                                        dblock_nz_.colptr(i_reaction),
                                        PETSC_NULL,
                                        oblock_nz_.colptr(i_reaction));
      CHKERRQ(ierr);
    } else if ((strcmp(mtype, MATAIJ) == 0) || (strcmp(mtype, MATMPIAIJ) == 0) || (strcmp(mtype, MATSEQAIJ) == 0)) {
      ierr = MatMPIAIJSetPreallocation(tv_mats_[i],
                                       PETSC_NULL,
                                       dblock_nz_.colptr(i_reaction),
                                       PETSC_NULL,
                                       oblock_nz_.colptr(i_reaction));
      CHKERRQ(ierr);
    }
    MatSetUp(tv_mats_[i]);
    
    new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], diag_vals_.colptr(i_reaction), prop_x_args);
    for (int i_state{0}; i_state < n_local_states; ++i_state) {
      // Set values for the diagonal block
      ierr = MatSetValue(tv_mats_[i],
                         own_start + i_state,
                         own_start + i_state,
                         -1.0 * diag_vals_(i_state, i_reaction),
                         INSERT_VALUES);
      CHKERRQ(ierr);
      ierr = MatSetValue(tv_mats_[i], own_start + i_state, offdiag_col_idxs_(i_state, i_reaction),
                         offdiag_vals_(i_state, i_reaction), INSERT_VALUES);
      CHKERRQ(ierr);
    }
    
    ierr = MatAssemblyBegin(tv_mats_[i], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tv_mats_[i], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }
  
  // Fill values for the time-invariant part
  if (!ti_reactions_.empty()) {
    // Determine the number of nonzeros on diagonal and offdiagonal blocks
    ti_dblock_nz_.set_size(num_rows_local_);
    ti_oblock_nz_.set_size(num_rows_local_);
    ti_dblock_nz_.fill(1 - ( int ) ti_reactions_.size());
    ti_oblock_nz_.fill(0);
    for (auto i_reaction: ti_reactions_) {
      ti_dblock_nz_ += dblock_nz_.col(i_reaction);
      ti_oblock_nz_ += oblock_nz_.col(i_reaction);
    }
    
    ierr = MatCreate(comm_, ti_mat_.mem());
    CHKERRQ(ierr);
    ierr = MatSetType(ti_mat_, MATMPISELL);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(ti_mat_);
    CHKERRQ(ierr);
    ierr = MatSetSizes(ti_mat_, num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_);
    CHKERRQ(ierr);
    ierr = MatGetType(ti_mat_, &mtype);
    CHKERRQ(ierr);
    if ((strcmp(mtype, MATSELL) == 0) || (strcmp(mtype, MATMPISELL) == 0) || (strcmp(mtype, MATSEQSELL) == 0)) {
      ierr = MatMPISELLSetPreallocation(ti_mat_, PETSC_NULL, &ti_dblock_nz_[0], PETSC_NULL, &ti_oblock_nz_[0]);
      CHKERRQ(ierr);
    } else if ((strcmp(mtype, MATAIJ) == 0) || (strcmp(mtype, MATMPIAIJ) == 0) || (strcmp(mtype, MATSEQAIJ) == 0)) {
      ierr = MatMPIAIJSetPreallocation(ti_mat_, PETSC_NULL, &ti_dblock_nz_[0], PETSC_NULL, &ti_oblock_nz_[0]);
      CHKERRQ(ierr);
    }
    MatSetUp(ti_mat_);
    
    for (auto i_reaction: ti_reactions_) {
      new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], diag_vals_.colptr(i_reaction), prop_x_args);
      for (int i_state{0}; i_state < n_local_states; ++i_state) {
        // Set values for the diagonal block
        ierr = MatSetValue(ti_mat_, own_start + i_state, own_start + i_state, -1.0 * diag_vals_(i_state, i_reaction),
                           ADD_VALUES);
        CHKERRQ(ierr);
        ierr = MatSetValue(ti_mat_, own_start + i_state, offdiag_col_idxs_(i_state, i_reaction),
                           offdiag_vals_(i_state, i_reaction), ADD_VALUES);
        CHKERRQ(ierr);
      }
    }
    
    ierr = MatAssemblyBegin(ti_mat_, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ti_mat_, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }
  has_values_ = PETSC_TRUE;
  return 0;
}

FspMatrixBase::~FspMatrixBase() {
  Destroy();
  comm_ = MPI_COMM_NULL;
}

int FspMatrixBase::Destroy() {
  PetscErrorCode ierr;
  tv_mats_.clear();
  enable_reactions_.clear();
  if (*ti_mat_.mem() != nullptr) {
    ierr = MatDestroy(ti_mat_.mem());
    CHKERRQ(ierr);
  }
  tv_reactions_.clear();
  ti_reactions_.clear();
  // TODO: "tv" and "ti" are too close. Find more distinguishing names.
  if (work_ != nullptr) {
    ierr = VecDestroy(work_.mem());
    CHKERRQ(ierr);
  }
  has_values_ = PETSC_FALSE;
  return 0;
}

int FspMatrixBase::DetermineLayout_(const StateSetBase &fsp) {
  PetscErrorCode ierr;
  try {
    num_rows_local_ = fsp.GetNumLocalStates();
    ierr = 0;
  }
  catch (std::runtime_error &ex) {
    ierr = -1;
  }
  PACMENSLCHKERRQ(ierr);
  
  // Generate matrix layout from Fsp's layout
  ierr = VecCreate(comm_, work_.mem());
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_);
  CHKERRQ(ierr);
  ierr = VecSetSizes(work_, num_rows_local_, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(work_);
  CHKERRQ(ierr);
  ierr = VecGetSize(work_, &num_rows_global_);
  CHKERRQ(ierr);
  return 0;
}

PacmenslErrorCode FspMatrixBase::SetTimeFun(TcoefFun new_t_fun, void *new_t_fun_args) {
  t_fun_ = new_t_fun;
  t_fun_args_ = new_t_fun_args;
  return 0;
}

int FspMatrixBase::CreateRHSJacobian(Mat *A) {
  int ierr;
  
  PetscInt own_start, own_end, num_local_states;
  arma::Col<Int> d_nz, o_nz;
  
  ierr = MatCreate(comm_, A);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);
  CHKERRQ(ierr);
  ierr = MatSetSizes(*A, num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_);
  CHKERRQ(ierr);
  ierr = MatSetUp(*A);
  CHKERRQ(ierr);
  
  num_local_states = offdiag_vals_.n_rows;
  d_nz.set_size(num_rows_local_);
  o_nz.set_size(num_rows_local_);
  d_nz.fill(1);
  o_nz.fill(0);
  for (auto ir: tv_reactions_) {
    for (int i{0}; i < num_local_states; ++i) {
      d_nz(i) += (dblock_nz_(i, ir) - 1);
      o_nz(i) += oblock_nz_(i, ir);
    }
  }
  if (!ti_reactions_.empty()) {
    d_nz += ti_dblock_nz_;
    o_nz += ti_oblock_nz_;
    d_nz -= 1;
  }
  
  MatMPIAIJSetPreallocation(*A, PETSC_NULL, &d_nz[0], PETSC_NULL, &o_nz[0]);
  
  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);
  for (auto ir: enable_reactions_) {
    for (int i{0}; i < num_local_states; ++i) {
      ierr = MatSetValue(*A, own_start + i, offdiag_col_idxs_(i, ir), 0.0, INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }
  for (int i{0}; i < num_rows_local_; ++i) {
    ierr = MatSetValue(*A, own_start + i, own_start + i, 0.0, INSERT_VALUES);
    CHKERRQ(ierr);
  }
  MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  return 0;
}

// If not in update mode, enter all values, if in update mode, only change the diagonal entries and the entries corresponding to
// time-varying reactions
int FspMatrixBase::ComputeRHSJacobian(PetscReal t, Mat A) {
  int ierr{0};
  PetscInt own_start, own_end, num_local_states;
  num_local_states = offdiag_vals_.n_rows;
  
  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  PACMENSLCHKERRQ(ierr);
  
  ierr = MatZeroEntries(A);
  PACMENSLCHKERRQ(ierr);
  
  
  // Update off-diagonal entries
  if (!tv_reactions_.empty()) {
    ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
    PACMENSLCHKERRQ(ierr);
    for (auto ir: tv_reactions_) {
      for (int i{0}; i < num_local_states; ++i) {
        ierr = MatSetValue(A,
                           own_start + i,
                           offdiag_col_idxs_(i, ir),
                           time_coefficients_[ir] * offdiag_vals_(i, ir),
                           ADD_VALUES);
        PACMENSLCHKERRQ(ierr);
      }
    }
  }
  
  if (!ti_reactions_.empty()) {
    for (auto ir: ti_reactions_) {
      
      for (int i{0}; i < num_local_states; ++i) {
        ierr = MatSetValue(A, own_start + i, offdiag_col_idxs_(i, ir), offdiag_vals_(i, ir), ADD_VALUES);
        PACMENSLCHKERRQ(ierr);
      }
    }
  }
  
  
  // Update the diagonal entries
  if (!tv_reactions_.empty()) {
    ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
    PACMENSLCHKERRQ(ierr);
    for (auto ir: tv_reactions_) {
      for (int i{0}; i < num_local_states; ++i) {
        ierr =
            MatSetValue(A, own_start + i, own_start + i, -1.0 * time_coefficients_[ir] * diag_vals_(i, ir), ADD_VALUES);
        PACMENSLCHKERRQ(ierr);
      }
    }
  }
  
  if (!ti_reactions_.empty()) {
    for (auto ir: ti_reactions_) {
      for (int i{0}; i < num_local_states; ++i) {
        ierr = MatSetValue(A, own_start + i, own_start + i, -1.0 * diag_vals_(i, ir), ADD_VALUES);
        PACMENSLCHKERRQ(ierr);
      }
    }
  }
  
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  PACMENSLCHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  PACMENSLCHKERRQ(ierr);
  return 0;
}

PacmenslErrorCode FspMatrixBase::GetLocalMVFlops(PetscInt *nflops) {
  PetscInt ierr;
  MatInfo minfo;
  
  ierr = MatGetInfo(ti_mat_, MAT_LOCAL, &minfo);
  CHKERRQ(ierr);
  *nflops = 2 * (( PetscInt ) minfo.nz_used);
  
  for (int j{0}; j < tv_mats_.size(); ++j) {
    ierr = MatGetInfo(tv_mats_[j], MAT_LOCAL, &minfo);
    CHKERRQ(ierr);
    *nflops += 2 * (( PetscInt ) minfo.nz_used) + num_rows_local_;
  }
  
  return 0;
}

}
