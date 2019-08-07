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
  ierr = FspMatrixBase::Action(t, x, y);
  PACMENSLCHKERRQ(ierr);
  ierr = VecGetLocalVector(x, xx);
  CHKERRQ(ierr);
  ierr = VecSet(sink_entries_, 0.0);
  CHKERRQ(ierr);
  for (int i : enable_reactions_)
  {
    ierr = MatMult(local_sinks_mat_[i], xx, sink_tmp);
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
  if (!local_sinks_mat_.empty())
  {
    for (int i{0}; i < local_sinks_mat_.size(); ++i)
    {
      if (local_sinks_mat_[i])
      {
        ierr = MatDestroy(&local_sinks_mat_[i]);
        CHKERRQ(ierr);
      }
    }
    local_sinks_mat_.clear();
  }
  if (sink_scatter_ctx_ != nullptr)
  {
    ierr = VecScatterDestroy(&sink_scatter_ctx_);
    CHKERRQ(ierr);
  }

  if (parallel_sink_mats_generated)
  {
    parallel_sinks_mat_.clear();
    parallel_sink_mats_generated = false;
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
                                                       const TcoefFun &new_prop_t,
                                                       const PropFun &prop,
                                                       const std::vector<int> &enable_reactions,
                                                       void *prop_t_args,
                                                       void *prop_args)
{
  PetscErrorCode ierr{0};

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
                                       new_prop_t,
                                       prop,
                                       enable_reactions,
                                       prop_t_args,
                                       prop_args);
  PACMENSLCHKERRQ(ierr);

  // Generate the extra blocks corresponding to sink states_
  const arma::Mat<int> &state_list    = constrained_fss_ptr->GetStatesRef();
  int                  n_local_states = constrained_fss_ptr->GetNumLocalStates();
  arma::Mat<int>       can_reach_my_state;

  sink_nnz.set_size(num_constraints_, num_reactions_);
  sink_nnz.zeros();
  sink_inz.resize(num_reactions_);
  sink_rows.resize(num_reactions_);

  for (auto ir : enable_reactions_)
  {
    sink_inz[ir].resize(num_constraints_);
    sink_rows[ir].resize(num_constraints_);
  }

  local_sinks_mat_.resize(num_reactions_);
  // Workspace for checking constraints
  arma::Mat<PetscInt> constraints_satisfied(n_local_states, num_constraints_);
  arma::Col<PetscInt> nconstraints_satisfied;
  for (auto           i_reaction : enable_reactions_)
  {
    // Count nnz for rows that represent sink states_
    can_reach_my_state = state_list + arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    ierr               = constrained_fss_ptr->CheckConstraints(n_local_states, can_reach_my_state.colptr(0),
                                                               constraints_satisfied.colptr(0));
    PACMENSLCHKERRQ(ierr);
    nconstraints_satisfied = arma::sum(constraints_satisfied, 1);

    for (int i_constr = 0; i_constr < num_constraints_; ++i_constr)
    {
      sink_nnz(i_constr, i_reaction) = n_local_states - arma::sum(constraints_satisfied.col(i_constr));
      sink_inz[i_reaction][i_constr].set_size(sink_nnz(i_constr, i_reaction));
      sink_rows[i_reaction][i_constr].set_size(sink_nnz(i_constr, i_reaction));
    }
    // Store the column indices and values of the nonzero entries on the sink rows
    for (int i_constr = 0; i_constr < num_constraints_; ++i_constr)
    {
      int      count   = 0;
      for (int i_state = 0; i_state < n_local_states; ++i_state)
      {
        if (constraints_satisfied(i_state, i_constr) == 0)
        {
          sink_inz[i_reaction][i_constr][count] = i_state;
          prop(i_reaction, state_list.n_rows, 1, state_list.colptr(i_state),
               &sink_rows[i_reaction][i_constr][count], prop_args);
          count += 1;
        }
      }
    }

    ierr = MatCreate(PETSC_COMM_SELF, &local_sinks_mat_[i_reaction]);
    CHKERRQ(ierr);
    ierr = MatSetType(local_sinks_mat_[i_reaction], MATSEQAIJ);
    CHKERRQ(ierr);
    ierr =
        MatSetSizes(local_sinks_mat_[i_reaction], num_constraints_, num_rows_local_, num_constraints_, num_rows_local_);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(local_sinks_mat_[i_reaction], num_constraints_ * n_local_states, NULL);
    CHKERRQ(ierr);

    for (auto i_constr{0}; i_constr < num_constraints_; i_constr++)
    {
      ierr = MatSetValues(local_sinks_mat_[i_reaction], 1, &i_constr, sink_nnz(i_constr, i_reaction),
                          sink_inz[i_reaction][i_constr].memptr(),
                          sink_rows[i_reaction][i_constr].memptr(), ADD_VALUES);
      CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(local_sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(local_sinks_mat_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }

  // Local vectors for computing sink entries
  ierr = VecCreateSeq(PETSC_COMM_SELF, num_constraints_, &sink_entries_);
  CHKERRQ(ierr);
  ierr = VecSetUp(sink_entries_);
  CHKERRQ(ierr);
  ierr = VecDuplicate(sink_entries_, &sink_tmp);
  CHKERRQ(ierr);
  ierr = VecSetUp(sink_tmp);
  CHKERRQ(ierr);

  // Scatter context for adding sink values
  sink_global_indices.resize(num_constraints_);
  for (int  i{0}; i < num_constraints_; ++i)
  {
    sink_global_indices[i] = constrained_fss_ptr->GetNumGlobalStates() + i;
  }
  Petsc<IS> sink_is;
  ierr = ISCreateGeneral(comm_, num_constraints_, sink_global_indices.data(), PETSC_COPY_VALUES, sink_is.mem());
  CHKERRQ(ierr);
  ierr = VecScatterCreate(sink_entries_, NULL, work_, sink_is, &sink_scatter_ctx_);
  CHKERRQ(ierr);
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
  ierr = VecCreate(comm_, &work_);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_);
  CHKERRQ(ierr);
  ierr = VecSetSizes(work_, num_rows_local_, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(work_);
  CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);
  return 0;
}

FspMatrixConstrained::FspMatrixConstrained(const FspMatrixConstrained &A) : FspMatrixBase(A)
{
  int ierr;
  num_constraints_ = A.num_constraints_;
  sinks_rank_      = A.sinks_rank_; ///< rank of the processor that stores sink states
  ierr             = VecDuplicate(A.sink_entries_, &sink_entries_);
  PETSCCHKERRTHROW(ierr);
  ierr = VecDuplicate(A.sink_tmp, &sink_tmp);
  PETSCCHKERRTHROW(ierr);
  ierr = VecScatterCopy(A.sink_scatter_ctx_, &sink_scatter_ctx_);
  PETSCCHKERRTHROW(ierr);

  local_sinks_mat_.resize(num_reactions_);
  for (int i = 0; i < num_reactions_; ++i)
  {
    ierr = MatDuplicate(A.local_sinks_mat_[i], MAT_COPY_VALUES, &local_sinks_mat_[i]);
    PETSCCHKERRTHROW(ierr);
  }

  sink_nnz = A.sink_nnz;
  sink_inz.resize(num_reactions_);
  sink_rows.resize(num_reactions_);
  for (int ir: enable_reactions_)
  {
    sink_inz[ir].resize(num_constraints_);
    sink_rows[ir].resize(num_constraints_);
    for (int ic = 0; ic < num_constraints_; ++ic)
    {
      sink_inz[ir][ic]  = A.sink_inz[ir][ic];
      sink_rows[ir][ic] = A.sink_rows[ir][ic];
    }
  }

  parallel_sink_mats_generated = A.parallel_sink_mats_generated;
  parallel_sinks_mat_.resize(A.parallel_sinks_mat_.size());
  if (parallel_sink_mats_generated)
  {
    for (int ir: enable_reactions_)
    {
      ierr = MatDuplicate(*A.parallel_sinks_mat_[ir].mem(), MAT_COPY_VALUES, parallel_sinks_mat_[ir].mem());
      PETSCCHKERRTHROW(ierr);
    }
  }
}

FspMatrixConstrained::FspMatrixConstrained(FspMatrixConstrained &&A) noexcept : FspMatrixBase(( FspMatrixBase && ) A)
{
  num_constraints_             = A.num_constraints_;
  sinks_rank_                  = A.sinks_rank_; ///< rank of the processor that stores sink states
  sink_entries_                = A.sink_entries_;
  sink_tmp                     = A.sink_tmp;
  sink_scatter_ctx_            = A.sink_scatter_ctx_;
  local_sinks_mat_             = std::move(A.local_sinks_mat_);
  parallel_sink_mats_generated = A.parallel_sink_mats_generated;
  parallel_sinks_mat_          = std::move(A.parallel_sinks_mat_);

  A.num_constraints_  = 0;
  A.sink_entries_     = nullptr;
  A.sink_tmp          = nullptr;
  A.sink_scatter_ctx_ = nullptr;
  A.local_sinks_mat_.clear();
  A.parallel_sink_mats_generated = false;

  sink_nnz  = std::move(A.sink_nnz);
  sink_inz  = std::move(A.sink_inz);
  sink_rows = std::move(A.sink_rows);
}

FspMatrixConstrained &FspMatrixConstrained::operator=(const FspMatrixConstrained &A)
{
  Destroy();
  FspMatrixBase::operator=(( const FspMatrixBase & ) A);
  int ierr;
  num_constraints_ = A.num_constraints_;
  sinks_rank_      = A.sinks_rank_; ///< rank of the processor that stores sink states

  ierr = VecDuplicate(A.sink_entries_, &sink_entries_);
  PETSCCHKERRTHROW(ierr);
  ierr = VecDuplicate(A.sink_tmp, &sink_tmp);
  PETSCCHKERRTHROW(ierr);
  ierr = VecScatterCopy(A.sink_scatter_ctx_, &sink_scatter_ctx_);
  PETSCCHKERRTHROW(ierr);

  local_sinks_mat_.resize(num_reactions_);
  for (int i = 0; i < num_reactions_; ++i)
  {
    ierr = MatDuplicate(A.local_sinks_mat_[i], MAT_COPY_VALUES, &local_sinks_mat_[i]);
    PETSCCHKERRTHROW(ierr);
  }

  parallel_sink_mats_generated = A.parallel_sink_mats_generated;
  parallel_sinks_mat_.resize(A.parallel_sinks_mat_.size());
  if (parallel_sink_mats_generated)
  {
    for (int ir: enable_reactions_)
    {
      ierr = MatDuplicate(*A.parallel_sinks_mat_[ir].mem(), MAT_COPY_VALUES, parallel_sinks_mat_[ir].mem());
      PETSCCHKERRTHROW(ierr);
    }
  }

  sink_nnz = A.sink_nnz;
  sink_inz.resize(num_reactions_);
  sink_rows.resize(num_reactions_);
  for (int ir: enable_reactions_)
  {
    sink_inz[ir].resize(num_constraints_);
    sink_rows[ir].resize(num_constraints_);
    for (int ic = 0; ic < num_constraints_; ++ic)
    {
      sink_inz[ir][ic]  = A.sink_inz[ir][ic];
      sink_rows[ir][ic] = A.sink_rows[ir][ic];
    }
  }

  return *this;
}

FspMatrixConstrained &FspMatrixConstrained::operator=(FspMatrixConstrained &&A) noexcept
{
  if (this != &A)
  {
    Destroy();
    FspMatrixBase::operator=(( FspMatrixBase && ) A);

    num_constraints_ = A.num_constraints_;
    sinks_rank_      = A.sinks_rank_; ///< rank of the processor that stores sink states

    sink_entries_                = A.sink_entries_;
    sink_tmp                     = A.sink_tmp;
    sink_scatter_ctx_            = A.sink_scatter_ctx_;
    local_sinks_mat_             = std::move(A.local_sinks_mat_);
    parallel_sink_mats_generated = A.parallel_sink_mats_generated;
    parallel_sinks_mat_          = std::move(A.parallel_sinks_mat_);

    A.num_constraints_  = 0;
    A.sink_entries_     = nullptr;
    A.sink_tmp          = nullptr;
    A.sink_scatter_ctx_ = nullptr;
    A.local_sinks_mat_.clear();
    A.parallel_sink_mats_generated = false;

    sink_nnz  = std::move(A.sink_nnz);
    sink_inz  = std::move(A.sink_inz);
    sink_rows = std::move(A.sink_rows);
  }
  return *this;
}

PacmenslErrorCode FspMatrixConstrained::GenerateParallelSinkMats()
{
  int     ierr;
  MatType mt;

  std::vector<int> d_nnz(num_rows_local_, 0), o_nnz(num_rows_local_, 0);
  // Create parallel sink matrices
  parallel_sinks_mat_.resize(num_reactions_);
  // Count the nonzeros on the sink rows
  std::vector<int> local_sink_nnz(num_constraints_, 0);

  if (rank_ != sinks_rank_)
  {
    for (int i = 0; i < num_constraints_; ++i)
    {
      for (int j = 0; j < num_reactions_; ++j)
      {
        local_sink_nnz[i] += sink_nnz(i, j);
      }
    }
  }

  MPI_Reduce(local_sink_nnz.data(), &o_nnz[num_rows_local_ - num_constraints_], num_constraints_,
             MPI_INT, MPI_SUM, sinks_rank_, comm_);

  if (rank_ == sinks_rank_)
  {
    for (int i = 0; i < num_constraints_; ++i)
    {
      for (int j = 0; j < num_reactions_; ++j)
      {
        d_nnz[num_rows_local_ - num_constraints_ + i] += sink_nnz(i, j);
      }
    }
  }

  parallel_sinks_mat_.resize(num_reactions_);
  for (int ir: enable_reactions_)
  {
    ierr = MatCreate(comm_, parallel_sinks_mat_[ir].mem());
    CHKERRQ(ierr);
    ierr = MatSetSizes(parallel_sinks_mat_[ir], num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_);
    CHKERRQ(ierr);
    ierr = MatSetType(parallel_sinks_mat_[ir], MATAIJ);
    CHKERRQ(ierr);
    // Preallocate space for sparse matrix format
    ierr = MatMPIAIJSetPreallocation(parallel_sinks_mat_[ir], PETSC_NULL, d_nnz.data(), PETSC_NULL, o_nnz.data());
    CHKERRQ(ierr);
    MatSetUp(parallel_sinks_mat_[ir]);

    for (int i = 0; i < num_constraints_; ++i)
    {
      for (int i1 = 0; i1 < sink_inz[ir][i].n_elem; ++i1)
      {
        ierr = MatSetValue(parallel_sinks_mat_[ir],
                           sink_global_indices[i],
                           own_start + sink_inz[ir][i][i1],
                           sink_rows[ir][i][i1],
                           INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
    ierr       = MatAssemblyBegin(parallel_sinks_mat_[ir], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }
  for (int ir: enable_reactions_)
  {
    ierr = MatAssemblyEnd(parallel_sinks_mat_[ir], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }

  parallel_sink_mats_generated = true;
  return 0;
}

int FspMatrixConstrained::CreateRHSJacobianBasic(Mat *A)
{
  PetscErrorCode ierr;
  MatType        mt;

  ierr = MatCreate(comm_, A);
  CHKERRQ(ierr);
  ierr = MatSetSizes(*A, num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_);
  CHKERRQ(ierr);
  ierr = MatSetType(*A, MATAIJ);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);
  CHKERRQ(ierr);
  ierr = MatGetType(*A, &mt);
  CHKERRQ(ierr);

  int              num_local_states = irnz_.n_rows;
  // Count the number of nonzeros on each local row
  std::vector<int> d_nnz(num_rows_local_, 0), o_nnz(num_rows_local_, 0);
  for (int         i                = 0; i < irnz_.n_rows; ++i)
  {
    for (auto j : enable_reactions_)
    {
      d_nnz[i] += (d_nnz_(i, j) - 1);
      o_nnz[i] += o_nnz_(i, j);
    }
    d_nnz[i] += 1;
  }

  // Count the nonzeros on the sink rows
  std::vector<int> local_sink_nnz(num_constraints_, 0);

  if (rank_ != sinks_rank_)
  {
    for (int i = 0; i < num_constraints_; ++i)
    {
      for (int j = 0; j < num_reactions_; ++j)
      {
        local_sink_nnz[i] += sink_nnz(i, j);
      }
    }
  }

  MPI_Reduce(local_sink_nnz.data(), &o_nnz[num_local_states], num_constraints_,
             MPI_INT, MPI_SUM, sinks_rank_, comm_);

  if (rank_ == sinks_rank_)
  {
    for (int i = 0; i < num_constraints_; ++i)
    {
      d_nnz[num_local_states + i] = 1;
      for (int j = 0; j < num_reactions_; ++j)
      {
        d_nnz[num_local_states + i] += sink_nnz(i, j);
      }
    }
  }

  // Preallocate space for sparse matrix format
  ierr = MatMPIAIJSetPreallocation(*A, PETSC_NULL, d_nnz.data(), PETSC_NULL, o_nnz.data());
  CHKERRQ(ierr);
  ierr = MatSetUp(*A);
  CHKERRQ(ierr);

  // Fill in the sparsity pattern with zeros
  for (int i = 0; i < num_rows_local_; ++i)
  {
    // Set values for the diagonal block
    ierr = MatSetValue(*A, own_start + i, own_start + i, 0.0,
                       INSERT_VALUES);
    CHKERRQ(ierr);
  }
  for (int ir: enable_reactions_)
  {
    for (int i = 0; i < num_local_states; ++i)
    {
      ierr = MatSetValue(*A, own_start + i, irnz_(i, ir),
                         0.0, INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }
  for (int ir: enable_reactions_)
  {
    for (int i = 0; i < num_constraints_; ++i)
    {
      for (int i1 = 0; i1 < sink_inz[ir][i].n_elem; ++i1)
      {
        ierr = MatSetValue(*A, sink_global_indices[i], own_start + sink_inz[ir][i][i1], 0.0, INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  if (!parallel_sink_mats_generated) GenerateParallelSinkMats();

  return 0;
}

int FspMatrixConstrained::ComputeRHSJacobianBasic(PetscReal t, Mat A)
{
  int       ierr;
  PetscBool assembled;
  ierr = MatAssembled(A, &assembled);
  CHKERRQ(ierr);
  if (!assembled) return -1;

  PetscPrintf(comm_, "Compute Jacobian \n");
  ierr = FspMatrixBase::ComputeRHSJacobianBasic(t, A);
  PACMENSLCHKERRQ(ierr);
  for (auto ir: enable_reactions_)
  {
    ierr = MatAXPY(A, time_coefficients_[ir], parallel_sinks_mat_[ir], SUBSET_NONZERO_PATTERN);
    CHKERRQ(ierr);
  }
  return 0;
}

int FspMatrixConstrained::CreateRHSJacobianCustom(Mat *A)
{
  return FspMatrixBase::CreateRHSJacobianCustom(A);
}

int FspMatrixConstrained::ComputeRHSJacoianCustom(Mat *A)
{
  return FspMatrixBase::ComputeRHSJacoianCustom(A);
}

}