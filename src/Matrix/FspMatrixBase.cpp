#include "Sys.h"
#include "FspMatrixBase.h"

namespace pacmensl {

FspMatrixBase::FspMatrixBase(MPI_Comm comm)
{
  MPI_Comm_dup(comm, &comm_);
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &comm_size_);
}

FspMatrixBase::FspMatrixBase(const FspMatrixBase &A)
{
  int ierr;
  ierr = MPI_Comm_dup(A.comm_, &comm_);
  MPICHKERRTHROW(ierr);
  rank_      = A.rank_;
  comm_size_ = A.comm_size_;

  t_fun_             = A.t_fun_;
  t_fun_args_        = A.t_fun_args_;
  time_coefficients_ = A.time_coefficients_;

  num_reactions_   = A.num_reactions_;
  num_rows_global_ = A.num_rows_global_;
  num_rows_local_  = A.num_rows_local_;
  enable_reactions_ = A.enable_reactions_;

  if (A.xx)
  {
    ierr = VecDuplicate(A.xx, &xx);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.yy)
  {
    ierr = VecDuplicate(A.yy, &yy);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.zz)
  {
    ierr = VecDuplicate(A.zz, &zz);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.work_)
  {
    ierr = VecDuplicate(A.work_, &work_);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.action_ctx_)
  {
    ierr = VecScatterCopy(A.action_ctx_, &action_ctx_);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.lvec_)
  {
    lvec_length_ = A.lvec_length_;
    ierr         = VecDuplicate(A.lvec_, &lvec_);
    PETSCCHKERRTHROW(ierr);
  }

  if (num_rows_global_)
  {
    diag_mats_.resize(num_reactions_);
    offdiag_mats_.resize(num_reactions_);
    for (int i{0}; i < num_reactions_; ++i)
    {
      ierr = MatDuplicate(*const_cast<Mat *>(A.diag_mats_[i].mem()), MAT_COPY_VALUES, diag_mats_[i].mem());
      PETSCCHKERRTHROW(ierr);
      ierr = MatDuplicate(*const_cast<Mat *>(A.offdiag_mats_[i].mem()), MAT_COPY_VALUES, offdiag_mats_[i].mem());
      PETSCCHKERRTHROW(ierr);
    }
  }
}

FspMatrixBase::FspMatrixBase(FspMatrixBase &&A) noexcept
{
  int ierr;

  Destroy();
  if (comm_) MPI_Comm_free(&comm_);

  comm_      = A.comm_;
  rank_      = A.rank_;
  comm_size_ = A.comm_size_;

  t_fun_             = A.t_fun_;
  t_fun_args_        = A.t_fun_args_;
  time_coefficients_ = A.time_coefficients_;

  num_reactions_   = A.num_reactions_;
  num_rows_global_ = A.num_rows_global_;
  num_rows_local_  = A.num_rows_local_;
  enable_reactions_ = A.enable_reactions_;

  lvec_length_     = A.lvec_length_;
  xx               = A.xx;
  yy               = A.yy;
  zz               = A.zz;
  work_            = A.work_;
  lvec_            = A.lvec_;
  action_ctx_      = A.action_ctx_;

  diag_mats_    = std::move(A.diag_mats_);
  offdiag_mats_ = std::move(A.offdiag_mats_);

  A.comm_       = nullptr;
  A.rank_       = 0;
  A.comm_size_  = 0;
  A.t_fun_      = nullptr;
  A.t_fun_args_ = nullptr;
  A.time_coefficients_.clear();
  A.num_reactions_ = 0;
  A.xx             = nullptr;
  A.yy             = nullptr;
  A.zz             = nullptr;
  A.work_          = nullptr;
  A.lvec_          = nullptr;
  A.action_ctx_    = nullptr;
  A.diag_mats_.clear();
  A.offdiag_mats_.clear();
}

FspMatrixBase &FspMatrixBase::operator=(const FspMatrixBase &A)
{
  int ierr;

  Destroy();
  if (comm_) MPI_Comm_free(&comm_);

  ierr = MPI_Comm_dup(A.comm_, &comm_);
  MPICHKERRTHROW(ierr);
  rank_      = A.rank_;
  comm_size_ = A.comm_size_;

  t_fun_             = A.t_fun_;
  t_fun_args_        = A.t_fun_args_;
  time_coefficients_ = A.time_coefficients_;

  num_reactions_   = A.num_reactions_;
  num_rows_global_ = A.num_rows_global_;
  num_rows_local_  = A.num_rows_local_;
  enable_reactions_ = A.enable_reactions_;

  if (A.xx)
  {
    ierr = VecDuplicate(A.xx, &xx);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.yy)
  {
    ierr = VecDuplicate(A.yy, &yy);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.zz)
  {
    ierr = VecDuplicate(A.zz, &zz);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.work_)
  {
    ierr = VecDuplicate(A.work_, &work_);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.action_ctx_)
  {
    ierr = VecScatterCopy(A.action_ctx_, &action_ctx_);
    PETSCCHKERRTHROW(ierr);
  }
  if (A.lvec_)
  {
    lvec_length_ = A.lvec_length_;
    ierr         = VecDuplicate(A.lvec_, &lvec_);
    PETSCCHKERRTHROW(ierr);
  }

  if (num_rows_global_)
  {
    diag_mats_.resize(num_reactions_);
    offdiag_mats_.resize(num_reactions_);
    for (int i{0}; i < num_reactions_; ++i)
    {
      ierr = MatDuplicate(*const_cast<Mat *>(A.diag_mats_[i].mem()), MAT_COPY_VALUES, diag_mats_[i].mem());
      PETSCCHKERRTHROW(ierr);
      ierr = MatDuplicate(*const_cast<Mat *>(A.offdiag_mats_[i].mem()), MAT_COPY_VALUES, offdiag_mats_[i].mem());
      PETSCCHKERRTHROW(ierr);
    }
  }

  return *this;
}

FspMatrixBase &FspMatrixBase::operator=(FspMatrixBase &&A) noexcept
{
  int ierr;

  if (this != &A)
  {
    Destroy();
    if (comm_) MPI_Comm_free(&comm_);

    comm_      = A.comm_;
    rank_      = A.rank_;
    comm_size_ = A.comm_size_;

    t_fun_             = A.t_fun_;
    t_fun_args_        = A.t_fun_args_;
    time_coefficients_ = A.time_coefficients_;

    num_reactions_   = A.num_reactions_;
    num_rows_global_ = A.num_rows_global_;
    num_rows_local_  = A.num_rows_local_;
    enable_reactions_ = A.enable_reactions_;

    lvec_length_     = A.lvec_length_;
    xx               = A.xx;
    yy               = A.yy;
    zz               = A.zz;
    work_            = A.work_;
    lvec_            = A.lvec_;
    action_ctx_      = A.action_ctx_;

    diag_mats_    = std::move(A.diag_mats_);
    offdiag_mats_ = std::move(A.offdiag_mats_);

    A.comm_       = nullptr;
    A.rank_       = 0;
    A.comm_size_  = 0;
    A.t_fun_      = nullptr;
    A.t_fun_args_ = nullptr;
    A.time_coefficients_.clear();
    A.num_reactions_ = 0;
    A.xx             = nullptr;
    A.yy             = nullptr;
    A.zz             = nullptr;
    A.work_          = nullptr;
    A.lvec_          = nullptr;
    A.action_ctx_    = nullptr;
    A.diag_mats_.clear();
    A.offdiag_mats_.clear();
  }

  return *this;
}

int FspMatrixBase::Action(PetscReal t, Vec x, Vec y)
{
  Int ierr;

  ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
  PACMENSLCHKERRQ(ierr);

  ierr = VecSet(y, 0.0);
  CHKERRQ(ierr);

  ierr = VecGetLocalVector(x, xx);
  CHKERRQ(ierr);
  ierr = VecGetLocalVector(y, yy);
  CHKERRQ(ierr);

  ierr = VecScatterBegin(action_ctx_, x, lvec_, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  for (auto ir : enable_reactions_)
  {
    ierr = MatMult(diag_mats_[ir], xx, zz);
    CHKERRQ(ierr);

    ierr = VecAXPY(yy, time_coefficients_[ir], zz);
    CHKERRQ(ierr);
  }
  ierr = VecScatterEnd(action_ctx_, x, lvec_, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  for (auto ir : enable_reactions_)
  {
    ierr = MatMult(offdiag_mats_[ir], lvec_, zz);
    CHKERRQ(ierr);

    ierr = VecAXPY(yy, time_coefficients_[ir], zz);
    CHKERRQ(ierr);
  }

  ierr = VecRestoreLocalVector(x, xx);
  CHKERRQ(ierr);
  ierr = VecRestoreLocalVector(y, yy);
  CHKERRQ(ierr);
  return 0;
}

PacmenslErrorCode FspMatrixBase::GenerateValues(const StateSetBase &fsp,
                                                const arma::Mat<Int> &SM,
                                                const TcoefFun &new_prop_t,
                                                const PropFun &new_prop_x,
                                                const std::vector<int> &enable_reactions,
                                                void *prop_t_args,
                                                void *prop_x_args)
{
  PacmenslErrorCode    ierr;
  PetscInt             n_species, n_local_states, own_start, own_end;
  const arma::Mat<Int> &state_list = fsp.GetStatesRef();
  arma::Mat<Int>       can_reach_my_state(state_list.n_rows, state_list.n_cols);

  ierr = DetermineLayout_(fsp);
  PACMENSLCHKERRQ(ierr);

  // Get the global number of rows
  ierr = VecGetSize(work_, &num_rows_global_);
  CHKERRQ(ierr);

  // arrays for counting nonzero entries on the diagonal and off-diagonal blocks
  arma::Mat<Int>       d_nnz, o_nnz;
  // global indices of off-processor entries needed for matrix-vector product
  arma::Row<Int>       out_indices;
  // arrays of nonzero column indices
  arma::Mat<Int>       irnz, irnz_off;
  // array o fmatrix values
  arma::Mat<PetscReal> mat_vals;

  ISLocalToGlobalMapping local2global_rows, local2global_lvec;

  n_species      = fsp.GetNumSpecies();
  n_local_states = fsp.GetNumLocalStates();
  num_reactions_ = fsp.GetNumReactions();
  t_fun_         = new_prop_t;
  t_fun_args_    = prop_t_args;
  diag_mats_.resize(num_reactions_);
  offdiag_mats_.resize(num_reactions_);
  time_coefficients_.set_size(num_reactions_);
  enable_reactions_ = enable_reactions;
  if (enable_reactions_.empty())
  {
    enable_reactions_ = std::vector<int>(num_reactions_);
    for (int i = 0; i < num_reactions_; ++i)
    {
      enable_reactions_[i] = i;
    }
  }

  MPI_Comm_rank(comm_, &rank_);

  // Find the nnz per row of diagonal and off-diagonal matrices
  irnz.set_size(n_local_states, num_reactions_);
  irnz_off.set_size(n_local_states, num_reactions_);
  out_indices.set_size(n_local_states * num_reactions_);
  mat_vals.set_size(n_local_states, num_reactions_);

  d_nnz.set_size(num_rows_local_, num_reactions_);
  o_nnz.set_size(num_rows_local_, num_reactions_);
  d_nnz.fill(1);
  o_nnz.zeros();
  int out_count = 0;
  irnz_off.fill(-1);

  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);
  // Count nnz for matrix rows
  for (auto i_reaction : enable_reactions_)
  {
    can_reach_my_state = state_list - arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    fsp.State2Index(can_reach_my_state, irnz.colptr(i_reaction));
    new_prop_x(i_reaction, can_reach_my_state.n_rows, can_reach_my_state.n_cols, &can_reach_my_state[0],
               mat_vals.colptr(i_reaction), prop_x_args);

    for (auto i_state{0}; i_state < n_local_states; ++i_state)
    {
      if (irnz(i_state, i_reaction) >= own_start && irnz(i_state, i_reaction) < own_end)
      {
        d_nnz(i_state, i_reaction) += 1;
      } else if (irnz(i_state, i_reaction) >= 0)
      {
        irnz_off(i_state, i_reaction) = irnz(i_state, i_reaction);
        irnz(i_state, i_reaction)     = -1;
        o_nnz(i_state, i_reaction) += 1;
        out_indices(out_count)        = irnz_off(i_state, i_reaction);
        out_count += 1;
      }
    }
  }

  // Create mapping from global rows to local rows
  PetscInt  *my_global_indices = new PetscInt[num_rows_local_];
  for (auto i{0}; i < num_rows_local_; ++i)
  {
    my_global_indices[i] = own_start + i;
  }
  ierr = ISLocalToGlobalMappingCreate(comm_, 1, num_rows_local_, my_global_indices, PETSC_COPY_VALUES,
                                      &local2global_rows);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(local2global_rows, ISLOCALTOGLOBALMAPPINGHASH);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetFromOptions(local2global_rows);
  CHKERRQ(ierr);
  delete[] my_global_indices;

  // Create mapping from local ghost vec to global indices
  out_indices.resize(out_count);
  arma::Row<Int> out_indices2 = arma::unique(out_indices);
  out_count = 0;
  for (auto i{0}; i < out_indices2.n_elem; ++i)
  {
    if (out_indices2[i] < own_start || out_indices2[i] >= own_end)
    {
      out_indices(out_count) = out_indices2[i];
      out_count += 1;
    }
  }

  lvec_length_ = PetscInt(out_count);
  ierr         = VecCreateSeq(PETSC_COMM_SELF, lvec_length_, &lvec_);
  CHKERRQ(ierr);
  ierr = VecSetUp(lvec_);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm_, 1, lvec_length_, &out_indices[0], PETSC_COPY_VALUES,
                                      &local2global_lvec);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(local2global_lvec, ISLOCALTOGLOBALMAPPINGHASH);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetFromOptions(local2global_lvec);
  CHKERRQ(ierr);

  // Create vecscatter for collecting off-diagonal vector entries
  IS from_is;
  ierr = ISCreateGeneral(comm_, lvec_length_, &out_indices[0], PETSC_COPY_VALUES, &from_is);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(work_, from_is, lvec_, PETSC_NULL, &action_ctx_);
  CHKERRQ(ierr);
  ierr = ISDestroy(&from_is);
  CHKERRQ(ierr);

  // Generate local vectors for matrix action
  VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &xx);
  VecSetUp(xx);
  VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &yy);
  VecSetUp(yy);
  VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &zz);
  VecSetUp(zz);

  // Convert the global indices of nonzero entries to local indices
  ierr = ISGlobalToLocalMappingApply(local2global_rows, IS_GTOLM_MASK, n_local_states * num_reactions_,
                                     irnz.memptr(), PETSC_NULL, irnz.memptr());
  CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(local2global_lvec, IS_GTOLM_MASK, n_local_states * num_reactions_,
                                     irnz_off.memptr(), PETSC_NULL, irnz_off.memptr());
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&local2global_lvec);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&local2global_rows);
  CHKERRQ(ierr);

  arma::Col<PetscReal> diag_vals(n_local_states);
  for (auto            i_reaction: enable_reactions_)
  {
    ierr = MatCreate(PETSC_COMM_SELF, diag_mats_[i_reaction].mem());
    CHKERRQ(ierr);
    ierr = MatSetType(diag_mats_[i_reaction], MATSEQAIJ);
    CHKERRQ(ierr);
    ierr = MatSetSizes(diag_mats_[i_reaction], num_rows_local_, num_rows_local_, num_rows_local_, num_rows_local_);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(diag_mats_[i_reaction], PETSC_NULL, d_nnz.colptr(i_reaction));
    CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF, offdiag_mats_[i_reaction].mem());
    CHKERRQ(ierr);
    ierr = MatSetType(offdiag_mats_[i_reaction], MATSEQAIJ);
    CHKERRQ(ierr);
    ierr = MatSetSizes(offdiag_mats_[i_reaction], num_rows_local_, lvec_length_, num_rows_local_, lvec_length_);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(offdiag_mats_[i_reaction], PETSC_NULL, o_nnz.colptr(i_reaction));
    CHKERRQ(ierr);

    new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], &diag_vals[0], prop_x_args);
    for (auto i_state{0}; i_state < n_local_states; ++i_state)
    {
      // Set values for the diagonal block
      ierr = MatSetValue(diag_mats_[i_reaction], i_state, i_state, -1.0 * diag_vals[i_state],
                         INSERT_VALUES);
      CHKERRQ(ierr);
      ierr = MatSetValue(diag_mats_[i_reaction], i_state, irnz(i_state, i_reaction),
                         mat_vals(i_state, i_reaction), INSERT_VALUES);
      CHKERRQ(ierr);

      // Set values for the off-diagonal block
      ierr = MatSetValue(offdiag_mats_[i_reaction], i_state, irnz_off(i_state, i_reaction),
                         mat_vals(i_state, i_reaction), INSERT_VALUES);
      CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(diag_mats_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(diag_mats_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyBegin(offdiag_mats_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(offdiag_mats_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }
  return 0;
}

FspMatrixBase::~FspMatrixBase()
{
  Destroy();
  if (comm_ != nullptr) MPI_Comm_free(&comm_);
}

int FspMatrixBase::Destroy()
{
  PetscErrorCode ierr;
  diag_mats_.clear();
  offdiag_mats_.clear();
  if (xx != nullptr)
  {
    ierr = VecDestroy(&xx);
    CHKERRQ(ierr);
  }
  if (yy != nullptr)
  {
    ierr = VecDestroy(&yy);
    CHKERRQ(ierr);
  }
  if (zz != nullptr)
  {
    ierr = VecDestroy(&zz);
    CHKERRQ(ierr);
  }
  if (work_ != nullptr)
  {
    ierr = VecDestroy(&work_);
    CHKERRQ(ierr);
  }
  if (lvec_ != nullptr)
  {
    ierr = VecDestroy(&lvec_);
    CHKERRQ(ierr);
  }
  if (action_ctx_ != nullptr)
  {
    ierr = VecScatterDestroy(&action_ctx_);
    CHKERRQ(ierr);
  }

  xx          = nullptr;
  yy          = nullptr;
  zz          = nullptr;
  work_       = nullptr;
  lvec_       = nullptr;
  action_ctx_ = nullptr;
  return 0;
}

PetscInt FspMatrixBase::GetLocalGhostLength() const
{
  return lvec_length_;
}

int FspMatrixBase::DetermineLayout_(const StateSetBase &fsp)
{
  PetscErrorCode ierr;
  try
  {
    num_rows_local_ = fsp.GetNumLocalStates();
    ierr            = 0;
  }
  catch (std::runtime_error &ex)
  {
    ierr = -1;
  }
  PACMENSLCHKERRQ(ierr);

  // Generate matrix layout from Fsp's layout
  ierr = VecCreate(comm_, &work_);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_);
  CHKERRQ(ierr);
  ierr = VecSetSizes(work_, num_rows_local_, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(work_);
  CHKERRQ(ierr);

  return 0;
}

PacmenslErrorCode FspMatrixBase::SetTimeFun(TcoefFun new_t_fun, void *new_t_fun_args)
{
  t_fun_ = new_t_fun;
  t_fun_args_ = new_t_fun_args;
  return 0;
}

}
