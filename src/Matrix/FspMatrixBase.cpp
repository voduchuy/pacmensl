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

  num_reactions_    = A.num_reactions_;
  num_rows_global_  = A.num_rows_global_;
  num_rows_local_   = A.num_rows_local_;
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

  irnz_        = A.irnz_;
  irnz_off_    = A.irnz_off_;
  out_indices_ = A.out_indices_;
  d_nnz_       = A.d_nnz_;
  o_nnz_       = A.o_nnz_;

  if (A.local2global_rows_ != nullptr)
  {
    ierr = ISLocalToGlobalMappingDuplicate(A.local2global_rows_, &local2global_rows_);
    PETSCCHKERRTHROW(ierr);
  }

  if (A.local2global_lvec_ != nullptr)
  {
    ierr = ISLocalToGlobalMappingDuplicate(A.local2global_lvec_, &local2global_lvec_);
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

  num_reactions_    = A.num_reactions_;
  num_rows_global_  = A.num_rows_global_;
  num_rows_local_   = A.num_rows_local_;
  enable_reactions_ = A.enable_reactions_;

  lvec_length_ = A.lvec_length_;
  xx           = A.xx;
  yy           = A.yy;
  zz           = A.zz;
  work_        = A.work_;
  lvec_        = A.lvec_;
  action_ctx_  = A.action_ctx_;

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

  irnz_              = std::move(A.irnz_);
  irnz_off_          = std::move(A.irnz_off_);
  out_indices_       = std::move(A.out_indices_);
  d_nnz_             = std::move(A.d_nnz_);
  o_nnz_             = std::move(A.o_nnz_);
  local2global_rows_ = A.local2global_rows_;
  local2global_lvec_ = A.local2global_lvec_;

  A.local2global_lvec_ = nullptr;
  A.local2global_rows_ = nullptr;
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

  num_reactions_    = A.num_reactions_;
  num_rows_global_  = A.num_rows_global_;
  num_rows_local_   = A.num_rows_local_;
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

  irnz_        = A.irnz_;
  irnz_off_    = A.irnz_off_;
  out_indices_ = A.out_indices_;
  d_nnz_       = A.d_nnz_;
  o_nnz_       = A.o_nnz_;

  if (A.local2global_rows_ != nullptr)
  {
    ierr = ISLocalToGlobalMappingDuplicate(A.local2global_rows_, &local2global_rows_);
    PETSCCHKERRTHROW(ierr);
  }

  if (A.local2global_lvec_ != nullptr)
  {
    ierr = ISLocalToGlobalMappingDuplicate(A.local2global_lvec_, &local2global_lvec_);
    PETSCCHKERRTHROW(ierr);
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

    num_reactions_    = A.num_reactions_;
    num_rows_global_  = A.num_rows_global_;
    num_rows_local_   = A.num_rows_local_;
    enable_reactions_ = A.enable_reactions_;

    lvec_length_ = A.lvec_length_;
    xx           = A.xx;
    yy           = A.yy;
    zz           = A.zz;
    work_        = A.work_;
    lvec_        = A.lvec_;
    action_ctx_  = A.action_ctx_;

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

    irnz_              = std::move(A.irnz_);
    irnz_off_          = std::move(A.irnz_off_);
    out_indices_       = std::move(A.out_indices_);
    d_nnz_             = std::move(A.d_nnz_);
    o_nnz_             = std::move(A.o_nnz_);
    local2global_rows_ = A.local2global_rows_;
    local2global_lvec_ = A.local2global_lvec_;

    A.local2global_lvec_ = nullptr;
    A.local2global_rows_ = nullptr;
  }

  return *this;
}

int FspMatrixBase::Action(PetscReal t, Vec x, Vec y)
{
  if (use_conventional_mats_)
  {
    return ActionBasic(t, x, y);
  } else
  {
    return ActionCustom(t, x, y);
  }
  return 0;
}

PacmenslErrorCode FspMatrixBase::ActionBasic(PetscReal t, Vec x, Vec y)
{
  PetscInt ierr;

  ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
  PACMENSLCHKERRQ(ierr);

  ierr = VecSet(y, 0.0);
  CHKERRQ(ierr);

  for (auto ir: enable_reactions_)
  {
    ierr = MatMult(diag_mats_[ir], x, work_);
    CHKERRQ(ierr);
    ierr = VecAXPY(y, time_coefficients_[ir], work_);
    CHKERRQ(ierr);
  }
  return 0;
}

PacmenslErrorCode FspMatrixBase::ActionCustom(PetscReal t, Vec x, Vec y)
{
  PetscInt ierr;

  ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
  PACMENSLCHKERRQ(ierr);

  ierr = VecSet(y, 0.0);
  CHKERRQ(ierr);

  ierr = VecGetLocalVectorRead(x, xx);
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

  ierr = VecRestoreLocalVectorRead(x, xx);
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
  if (!use_conventional_mats_)
  {
    return GenerateValuesCustom(fsp, SM, new_prop_t, new_prop_x, enable_reactions, prop_t_args, prop_x_args);
  } else
  {
    return GenerateValuesBasic(fsp, SM, new_prop_t, new_prop_x, enable_reactions, prop_t_args, prop_x_args);
  }
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

  if (local2global_lvec_)
  {
    ierr = ISLocalToGlobalMappingDestroy(&local2global_lvec_);
    CHKERRQ(ierr);
  }
  if (local2global_rows_)
  {
    ierr = ISLocalToGlobalMappingDestroy(&local2global_rows_);
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

  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);

  return 0;
}

PacmenslErrorCode FspMatrixBase::SetTimeFun(TcoefFun new_t_fun, void *new_t_fun_args)
{
  t_fun_      = new_t_fun;
  t_fun_args_ = new_t_fun_args;
  return 0;
}

int FspMatrixBase::CreateRHSJacobian(Mat *A)
{
  if (use_conventional_mats_)
  {
    return CreateRHSJacobianBasic(A);
  }
  return 0;
}

int FspMatrixBase::ComputeRHSJacobian(PetscReal t, Mat A)
{
  if (use_conventional_mats_)
  {
    return ComputeRHSJacobianBasic(t, A);
  }
  return 0;
}

PacmenslErrorCode FspMatrixBase::GenerateValuesBasic(const StateSetBase &fsp,
                                                     const arma::Mat<Int> &SM,
                                                     const TcoefFun &new_prop_t,
                                                     const PropFun &new_prop_x,
                                                     const std::vector<int> &enable_reactions,
                                                     void *prop_t_args,
                                                     void *prop_x_args)
{
  PacmenslErrorCode    ierr;
  PetscInt             n_species, n_local_states;
  const arma::Mat<Int> &state_list = fsp.GetStatesRef();
  arma::Mat<Int>       can_reach_my_state(state_list.n_rows, state_list.n_cols);

  ierr = DetermineLayout_(fsp);
  PACMENSLCHKERRQ(ierr);

  // Get the global number of rows
  ierr = VecGetSize(work_, &num_rows_global_);
  CHKERRQ(ierr);

  // array o fmatrix values
  arma::Mat<PetscReal> mat_vals;

  n_species      = fsp.GetNumSpecies();
  n_local_states = fsp.GetNumLocalStates();
  num_reactions_ = fsp.GetNumReactions();
  t_fun_         = new_prop_t;
  t_fun_args_    = prop_t_args;
  diag_mats_.resize(num_reactions_);
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
  irnz_.set_size(n_local_states, num_reactions_);
  irnz_off_.set_size(n_local_states, num_reactions_);
  out_indices_.set_size(n_local_states * num_reactions_);
  mat_vals.set_size(n_local_states, num_reactions_);

  d_nnz_.set_size(num_rows_local_, num_reactions_);
  o_nnz_.set_size(num_rows_local_, num_reactions_);
  d_nnz_.fill(1);
  o_nnz_.zeros();
  int out_count = 0;
  irnz_off_.fill(-1);

  ierr = VecGetOwnershipRange(work_, &own_start, &own_end);
  CHKERRQ(ierr);
  // Count nnz for matrix rows
  for (auto i_reaction : enable_reactions_)
  {
    can_reach_my_state = state_list - arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    fsp.State2Index(can_reach_my_state, irnz_.colptr(i_reaction));
    new_prop_x(i_reaction, can_reach_my_state.n_rows, can_reach_my_state.n_cols, &can_reach_my_state[0],
               mat_vals.colptr(i_reaction), prop_x_args);

    for (auto i_state{0}; i_state < n_local_states; ++i_state)
    {
      if (irnz_(i_state, i_reaction) >= own_start && irnz_(i_state, i_reaction) < own_end)
      {
        d_nnz_(i_state, i_reaction) += 1;
      } else if (irnz_(i_state, i_reaction) >= 0)
      {
        o_nnz_(i_state, i_reaction) += 1;
      }
    }
  }

  arma::Col<PetscReal> diag_vals(n_local_states);
  MatType              mtype;
  for (auto            i_reaction: enable_reactions_)
  {
    ierr = MatCreate(comm_, diag_mats_[i_reaction].mem());
    CHKERRQ(ierr);
    ierr = MatSetType(diag_mats_[i_reaction], MATAIJ);
    CHKERRQ(ierr);
    ierr = MatSetSizes(diag_mats_[i_reaction], num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_);
    CHKERRQ(ierr);

    ierr = MatMPIAIJSetPreallocation(diag_mats_[i_reaction],
                                     PETSC_NULL,
                                     d_nnz_.colptr(i_reaction),
                                     PETSC_NULL,
                                     o_nnz_.colptr(i_reaction));
    CHKERRQ(ierr);

    ierr = MatSetUp(diag_mats_[i_reaction]); CHKERRQ(ierr);

    new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], &diag_vals[0], prop_x_args);
    for (int i_state{0}; i_state < n_local_states; ++i_state)
    {
      // Set values for the diagonal block
      ierr = MatSetValue(diag_mats_[i_reaction], own_start + i_state, own_start + i_state, -1.0 * diag_vals[i_state],
                         INSERT_VALUES);
      CHKERRQ(ierr);
      ierr = MatSetValue(diag_mats_[i_reaction], own_start + i_state, irnz_(i_state, i_reaction),
                         mat_vals(i_state, i_reaction), INSERT_VALUES);
      CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(diag_mats_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(diag_mats_[i_reaction], MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
  }

  // Generate local vectors for matrix action
  VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &xx);
  VecSetUp(xx);
  VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &yy);
  VecSetUp(yy);
  VecCreateSeq(PETSC_COMM_SELF, num_rows_local_, &zz);
  VecSetUp(zz);

  return 0;
}

PacmenslErrorCode FspMatrixBase::GenerateValuesCustom(const StateSetBase &fsp,
                                                      const arma::Mat<Int> &SM,
                                                      const TcoefFun &new_prop_t,
                                                      const PropFun &new_prop_x,
                                                      const std::vector<int> &enable_reactions,
                                                      void *prop_t_args,
                                                      void *prop_x_args)
{
  PacmenslErrorCode    ierr;
  PetscInt             n_species, n_local_states;
  const arma::Mat<Int> &state_list = fsp.GetStatesRef();
  arma::Mat<Int>       can_reach_my_state(state_list.n_rows, state_list.n_cols);

  ierr = DetermineLayout_(fsp);
  PACMENSLCHKERRQ(ierr);

  // Get the global number of rows
  ierr = VecGetSize(work_, &num_rows_global_);
  CHKERRQ(ierr);

  // array o fmatrix values
  arma::Mat<PetscReal> mat_vals;

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
  irnz_.set_size(n_local_states, num_reactions_);
  irnz_off_.set_size(n_local_states, num_reactions_);
  out_indices_.set_size(n_local_states * num_reactions_);
  mat_vals.set_size(n_local_states, num_reactions_);

  d_nnz_.set_size(num_rows_local_, num_reactions_);
  o_nnz_.set_size(num_rows_local_, num_reactions_);
  d_nnz_.fill(1);
  o_nnz_.zeros();
  int out_count = 0;
  irnz_off_.fill(-1);

  // Count nnz for matrix rows
  for (auto i_reaction : enable_reactions_)
  {
    can_reach_my_state = state_list - arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
    fsp.State2Index(can_reach_my_state, irnz_.colptr(i_reaction));
    new_prop_x(i_reaction, can_reach_my_state.n_rows, can_reach_my_state.n_cols, &can_reach_my_state[0],
               mat_vals.colptr(i_reaction), prop_x_args);

    for (auto i_state{0}; i_state < n_local_states; ++i_state)
    {
      if (irnz_(i_state, i_reaction) >= own_start && irnz_(i_state, i_reaction) < own_end)
      {
        d_nnz_(i_state, i_reaction) += 1;
      } else if (irnz_(i_state, i_reaction) >= 0)
      {
        irnz_off_(i_state, i_reaction) = irnz_(i_state, i_reaction);
        irnz_(i_state, i_reaction)     = -1;
        o_nnz_(i_state, i_reaction) += 1;
        out_indices_(out_count)        = irnz_off_(i_state, i_reaction);
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
                                      &local2global_rows_);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(local2global_rows_, ISLOCALTOGLOBALMAPPINGHASH);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetFromOptions(local2global_rows_);
  CHKERRQ(ierr);
  delete[] my_global_indices;

  // Create mapping from local ghost vec to global indices
  out_indices_.resize(out_count);
  arma::Row<Int> out_indices2 = arma::unique(out_indices_);
  out_count = 0;
  for (auto i{0}; i < out_indices2.n_elem; ++i)
  {
    if (out_indices2[i] < own_start || out_indices2[i] >= own_end)
    {
      out_indices_(out_count) = out_indices2[i];
      out_count += 1;
    }
  }

  lvec_length_ = PetscInt(out_count);
  ierr         = VecCreateSeq(PETSC_COMM_SELF, lvec_length_, &lvec_);
  CHKERRQ(ierr);
  ierr = VecSetUp(lvec_);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm_, 1, lvec_length_, &out_indices_[0], PETSC_COPY_VALUES,
                                      &local2global_lvec_);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(local2global_lvec_, ISLOCALTOGLOBALMAPPINGHASH);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetFromOptions(local2global_lvec_);
  CHKERRQ(ierr);

  // Create vecscatter for collecting off-diagonal vector entries
  IS from_is;
  ierr = ISCreateGeneral(comm_, lvec_length_, &out_indices_[0], PETSC_COPY_VALUES, &from_is);
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
  ierr = ISGlobalToLocalMappingApply(local2global_rows_, IS_GTOLM_MASK, n_local_states * num_reactions_,
                                     irnz_.memptr(), PETSC_NULL, irnz_.memptr());
  CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(local2global_lvec_, IS_GTOLM_MASK, n_local_states * num_reactions_,
                                     irnz_off_.memptr(), PETSC_NULL, irnz_off_.memptr());
  CHKERRQ(ierr);

  arma::Col<PetscReal> diag_vals(n_local_states);
  MatType              mtype;
  for (auto            i_reaction: enable_reactions_)
  {
    ierr = MatCreate(PETSC_COMM_SELF, diag_mats_[i_reaction].mem());
    CHKERRQ(ierr);
    ierr = MatSetType(diag_mats_[i_reaction], MATSELL);
    CHKERRQ(ierr);
    ierr = MatSetSizes(diag_mats_[i_reaction], num_rows_local_, num_rows_local_, num_rows_local_, num_rows_local_);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(diag_mats_[i_reaction]);
    CHKERRQ(ierr);
    ierr = MatGetType(diag_mats_[i_reaction], &mtype);
    CHKERRQ(ierr);
    if ((strcmp(mtype, MATSELL) == 0) || (strcmp(mtype, MATMPISELL) == 0) || (strcmp(mtype, MATSEQSELL)) == 0)
    {
      ierr = MatSeqSELLSetPreallocation(diag_mats_[i_reaction], PETSC_NULL, d_nnz_.colptr(i_reaction));
      CHKERRQ(ierr);
    } else if ((strcmp(mtype, MATAIJ) == 0) || (strcmp(mtype, MATMPIAIJ) == 0) || (strcmp(mtype, MATSEQAIJ) == 0))
    {
      ierr = MatSeqAIJSetPreallocation(diag_mats_[i_reaction], PETSC_NULL, d_nnz_.colptr(i_reaction));
      CHKERRQ(ierr);
    }

    ierr = MatCreate(PETSC_COMM_SELF, offdiag_mats_[i_reaction].mem());
    CHKERRQ(ierr);
    ierr = MatSetType(offdiag_mats_[i_reaction], MATSEQSELL);
    CHKERRQ(ierr);
    ierr = MatSetSizes(offdiag_mats_[i_reaction], num_rows_local_, lvec_length_, num_rows_local_, lvec_length_);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(offdiag_mats_[i_reaction]);
    CHKERRQ(ierr);
    ierr = MatGetType(offdiag_mats_[i_reaction], &mtype);
    CHKERRQ(ierr);
    if ((strcmp(mtype, MATSELL) == 0) || (strcmp(mtype, MATMPISELL) == 0) || (strcmp(mtype, MATSEQSELL)) == 0)
    {
      ierr = MatSeqSELLSetPreallocation(offdiag_mats_[i_reaction], PETSC_NULL, o_nnz_.colptr(i_reaction));
      CHKERRQ(ierr);
    } else if (strcmp(mtype, MATAIJ) == 0 || strcmp(mtype, MATMPIAIJ) == 0 || strcmp(mtype, MATSEQAIJ) == 0)
    {
      ierr = MatSeqAIJSetPreallocation(offdiag_mats_[i_reaction], PETSC_NULL, o_nnz_.colptr(i_reaction));
      CHKERRQ(ierr);
    }

    new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], &diag_vals[0], prop_x_args);
    for (auto i_state{0}; i_state < n_local_states; ++i_state)
    {
      // Set values for the diagonal block
      ierr = MatSetValue(diag_mats_[i_reaction], i_state, i_state, -1.0 * diag_vals[i_state],
                         INSERT_VALUES);
      CHKERRQ(ierr);
      ierr = MatSetValue(diag_mats_[i_reaction], i_state, irnz_(i_state, i_reaction),
                         mat_vals(i_state, i_reaction), INSERT_VALUES);
      CHKERRQ(ierr);

      // Set values for the off-diagonal block
      ierr = MatSetValue(offdiag_mats_[i_reaction], i_state, irnz_off_(i_state, i_reaction),
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

PacmenslErrorCode FspMatrixBase::SetUseConventionalMats()
{
  use_conventional_mats_ = true;
  return 0;
}

int FspMatrixBase::CreateRHSJacobianBasic(Mat *A)
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
  ierr               = MatGetType(*A, &mt);
  CHKERRQ(ierr);

  // Count the number of nonzeros on each local row
  std::vector<int> d_nnz(num_rows_local_, 0), o_nnz(num_rows_local_, 0);
  for (int         i = 0; i < irnz_.n_rows; ++i)
  {
    for (auto j : enable_reactions_)
    {
      d_nnz[i] += (d_nnz_(i, j) - 1);
      o_nnz[i] += o_nnz_(i, j);
    }
    d_nnz[i] += 1;
  }

  ierr = MatMPIAIJSetPreallocation(*A, PETSC_NULL, d_nnz.data(), PETSC_NULL, o_nnz.data());
  CHKERRQ(ierr);

  MatSetUp(*A);

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
    for (int i = 0; i < num_rows_local_; ++i)
    {
      ierr = MatSetValue(*A, own_start + i, irnz_(i, ir),
                         0.0, INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  return 0;
}

int FspMatrixBase::ComputeRHSJacobianBasic(PetscReal t, Mat A)
{
  PetscInt ierr;

  ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_);
  PACMENSLCHKERRQ(ierr);

  ierr = MatZeroEntries(A);
  CHKERRQ(ierr);

  for (int ir: enable_reactions_)
  {
    ierr = MatAXPY(A, time_coefficients_[ir], diag_mats_[ir], SUBSET_NONZERO_PATTERN);
    CHKERRQ(ierr);
  }
  return 0;
}

int FspMatrixBase::CreateRHSJacobianCustom(Mat *A)
{
  return 0;
}

int FspMatrixBase::ComputeRHSJacoianCustom(Mat *A)
{
  return 0;
}

}
