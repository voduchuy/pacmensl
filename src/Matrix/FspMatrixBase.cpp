#include "Sys.h"
#include "FspMatrixBase.h"

namespace pacmensl {

FspMatrixBase::FspMatrixBase(MPI_Comm comm)
{
  comm_ = comm;
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &comm_size_);
}

int FspMatrixBase::Action(PetscReal t, Vec x, Vec y)
{
  PetscInt ierr;

  ierr = VecSet(y, 0.0); CHKERRQ(ierr);
  if (!tv_reactions_.empty()){
    ierr = t_fun_(t, num_reactions_, time_coefficients_.memptr(), t_fun_args_); PACMENSLCHKERRQ(ierr);

    for (auto ir: tv_reactions_){
      ierr = MatMult(tv_mats_[ir], x, work_); CHKERRQ(ierr);
      ierr = VecAXPY(y, time_coefficients_[ir], work_); CHKERRQ(ierr);
    }
  }

  if (ti_mat_.mem() != nullptr)
  {
    ierr = MatMult(ti_mat_, x, work_); CHKERRQ(ierr);
    ierr = VecAXPY(y, 1.0, work_); CHKERRQ(ierr);
  }
  return 0;
}

PacmenslErrorCode FspMatrixBase::GenerateValues(const StateSetBase &fsp,
                                                const arma::Mat<Int> &SM,
                                                std::vector<int> time_vayring,
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

  ierr = DetermineLayout_(fsp); PACMENSLCHKERRQ(ierr);

  // Get the global number of rows
  ierr = VecGetSize(work_, &num_rows_global_); CHKERRQ(ierr);

  // arrays for counting nonzero entries on the diagonal and off-diagonal blocks
  arma::Mat<Int>       d_nnz, o_nnz;
  // global indices of off-processor entries needed for matrix-vector product
  arma::Row<Int>       out_indices;
  // arrays of nonzero column indices
  arma::Mat<Int>       irnz, irnz_off;
  // array o fmatrix values
  arma::Mat<PetscReal> mat_vals;

  n_species      = fsp.GetNumSpecies();
  n_local_states = fsp.GetNumLocalStates();
  num_reactions_ = fsp.GetNumReactions();
  time_coefficients_.set_size(num_reactions_);

  t_fun_         = new_prop_t;
  t_fun_args_    = prop_t_args;
  tv_mats_.resize(num_reactions_);
  enable_reactions_ = enable_reactions;
  if (enable_reactions_.empty())
  {
    enable_reactions_ = std::vector<int>(num_reactions_);
    for (int i = 0; i < num_reactions_; ++i)
    {
      enable_reactions_[i] = i;
    }
  }
  for (int ir: enable_reactions_){
    if (std::find(time_vayring.begin(), time_vayring.end(), ir) != time_vayring.end()){
      tv_reactions_.push_back(ir);
    }
    else{
      ti_reactions_.push_back(ir);
    }
  }


  // Find the nnz per row of diagonal and off-diagonal matrices
  irnz.set_size(n_local_states, num_reactions_);
  irnz_off.set_size(n_local_states, num_reactions_);
  out_indices.set_size(n_local_states * num_reactions_);
  mat_vals.set_size(n_local_states, num_reactions_);

  d_nnz.set_size(num_rows_local_, num_reactions_);
  o_nnz.set_size(num_rows_local_, num_reactions_);
  d_nnz.fill(1);
  o_nnz.zeros();
  irnz_off.fill(-1);

  ierr = VecGetOwnershipRange(work_, &own_start, &own_end); CHKERRQ(ierr);
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
        o_nnz(i_state, i_reaction) += 1;
      }
    }
  }

  // Fill values for the time-varying part
  arma::Col<PetscReal> diag_vals(n_local_states);
  ierr = VecGetOwnershipRange(work_, &own_start, &own_end); CHKERRQ(ierr);
  MatType mtype;
  for (auto            i_reaction: tv_reactions_)
  {
    ierr = MatCreate(comm_, tv_mats_[i_reaction].mem()); CHKERRQ(ierr);
    ierr = MatSetType(tv_mats_[i_reaction], MATMPISELL); CHKERRQ(ierr);
    ierr = MatSetFromOptions(tv_mats_[i_reaction]); CHKERRQ(ierr);
    ierr = MatSetSizes(tv_mats_[i_reaction], num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_); CHKERRQ(ierr);
    ierr = MatGetType(tv_mats_[i_reaction], &mtype); CHKERRQ(ierr);
    if ( (strcmp(mtype, MATSELL) == 0 )|| (strcmp(mtype, MATMPISELL) == 0 ) || (strcmp(mtype, MATSEQSELL) == 0)){
      ierr = MatMPISELLSetPreallocation(tv_mats_[i_reaction], PETSC_NULL, d_nnz.colptr(i_reaction), PETSC_NULL, o_nnz.colptr(i_reaction)); CHKERRQ(ierr);
    }
    else if ((strcmp(mtype, MATAIJ) == 0 )|| (strcmp(mtype, MATMPIAIJ) == 0) || (strcmp(mtype, MATSEQAIJ) == 0)){
      ierr = MatMPIAIJSetPreallocation(tv_mats_[i_reaction], PETSC_NULL, d_nnz.colptr(i_reaction), PETSC_NULL, o_nnz.colptr(i_reaction)); CHKERRQ(ierr);
    }
    MatSetUp(tv_mats_[i_reaction]);

    new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], &diag_vals[0], prop_x_args);
    for (int i_state{0}; i_state < n_local_states; ++i_state)
    {
      // Set values for the diagonal block
      ierr = MatSetValue(tv_mats_[i_reaction], own_start + i_state, own_start + i_state, -1.0 * diag_vals[i_state],
                         INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(tv_mats_[i_reaction], own_start + i_state, irnz(i_state, i_reaction),
                         mat_vals(i_state, i_reaction), INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(tv_mats_[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tv_mats_[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Fill values for the time-invariant part
  if (~ti_reactions_.empty()){
    // Determine the number of nonzeros on diagonal and offdiagonal blocks
    arma::Col<int> dnnz_ti(num_rows_local_), onnz_ti(num_rows_local_);
    dnnz_ti.fill(1 - (int) ti_reactions_.size());
    onnz_ti.fill(0);
    for (auto i_reaction: ti_reactions_){
      dnnz_ti += d_nnz.col(i_reaction);
      onnz_ti += o_nnz.col(i_reaction);
    }

    ierr = MatCreate(comm_, ti_mat_.mem()); CHKERRQ(ierr);
    ierr = MatSetType(ti_mat_, MATMPISELL); CHKERRQ(ierr);
    ierr = MatSetFromOptions(ti_mat_); CHKERRQ(ierr);
    ierr = MatSetSizes(ti_mat_, num_rows_local_, num_rows_local_, num_rows_global_, num_rows_global_); CHKERRQ(ierr);
    ierr = MatGetType(ti_mat_, &mtype); CHKERRQ(ierr);
    if ( (strcmp(mtype, MATSELL) == 0 )|| (strcmp(mtype, MATMPISELL) == 0 ) || (strcmp(mtype, MATSEQSELL) == 0)){
      ierr = MatMPISELLSetPreallocation(ti_mat_, PETSC_NULL, &dnnz_ti[0], PETSC_NULL, &onnz_ti[0]); CHKERRQ(ierr);
    }
    else if ((strcmp(mtype, MATAIJ) == 0 )|| (strcmp(mtype, MATMPIAIJ) == 0) || (strcmp(mtype, MATSEQAIJ) == 0)){
      ierr = MatMPIAIJSetPreallocation(ti_mat_, PETSC_NULL, &dnnz_ti[0], PETSC_NULL, &onnz_ti[0]); CHKERRQ(ierr);
    }
    MatSetUp(ti_mat_);

    for (auto i_reaction: ti_reactions_){
      new_prop_x(i_reaction, n_species, n_local_states, &state_list[0], &diag_vals[0], prop_x_args);
      for (int i_state{0}; i_state < n_local_states; ++i_state)
      {
        // Set values for the diagonal block
        ierr = MatSetValue(ti_mat_, own_start + i_state, own_start + i_state, -1.0 * diag_vals[i_state],
                           ADD_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(ti_mat_, own_start + i_state, irnz(i_state, i_reaction),
                           mat_vals(i_state, i_reaction), ADD_VALUES); CHKERRQ(ierr);
      }
    }

    ierr = MatAssemblyBegin(ti_mat_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ti_mat_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  return 0;
}

FspMatrixBase::~FspMatrixBase()
{
  Destroy();
  comm_ = nullptr;
}

int FspMatrixBase::Destroy()
{
  PetscErrorCode ierr;
  tv_mats_.clear();
  enable_reactions_.clear();
  MatDestroy(ti_mat_.mem());
  tv_reactions_.clear();
  ti_reactions_.clear();
  if (work_ != nullptr)
  {
    ierr = VecDestroy(work_.mem()); CHKERRQ(ierr);
  }

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
  } PACMENSLCHKERRQ(ierr);

  // Generate matrix layout from Fsp's layout
  ierr = VecCreate(comm_, work_.mem()); CHKERRQ(ierr);
  ierr = VecSetFromOptions(work_); CHKERRQ(ierr);
  ierr = VecSetSizes(work_, num_rows_local_, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(work_); CHKERRQ(ierr);

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

  return 0;
}

int FspMatrixBase::ComputeRHSJacobian(PetscReal t, Mat A)
{
  int       ierr;
  PetscBool created;
  return 0;
}

}
