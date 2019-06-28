//
// Created by Huy Vo on 2019-06-25.
//

#include "StationaryMCSolver.h"

pacmensl::StationaryMCSolver::StationaryMCSolver(MPI_Comm comm) {
  int ierr;
  ierr = MPI_Comm_dup(comm, &comm_);
  PACMENSLCHKERREXCEPT(ierr);
}

int pacmensl::StationaryMCSolver::SetSolutionVec(Vec *vec) {
  solution_ = vec;
  return 0;
}

int pacmensl::StationaryMCSolver::SetMatDiagonal(Vec *diag) {
  mat_diagonal_ = diag;
  return 0;
}

int pacmensl::StationaryMCSolver::SetUp() {
  int ierr;
  // Create Mat shell
  ierr = VecGetLocalSize(*solution_, &n_local_);
  CHKERRQ(ierr);
  ierr = VecGetSize(*solution_, &n_global_);
  CHKERRQ(ierr);
  ierr = MatCreateShell(comm_, n_local_, n_local_, n_global_, n_global_, ( void * ) &this->matvec_,
                        &inf_generator_);
  CHKERRQ(ierr);
  ierr = MatShellSetOperation(inf_generator_, MATOP_MAT_MULT, ( void (*)(void)) &ModifiedMatrixAction);
  CHKERRQ(ierr);
  // Create Krylov solver object
  ierr = KSPCreate(comm_, &ksp_);
  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_, inf_generator_, PETSC_NULL);
  CHKERRQ(ierr);
  return 0;
}

int pacmensl::StationaryMCSolver::SetMatVec(pacmensl::TIMatvec matvec) {
  matvec_ = matvec;
  return 0;
}

/**
 * @brief Given the infinitesimal generator A of the discrete Markov process, compute the action (A + (2/n)*d*q^T)*v
 * where n is the number of rows of A, d = diag(A), q = (1,1,..,1)^T.
 * @details Collective.
 * @param A infinitesimal generator of the Markov chain.
 * @param x input vector.
 * @param y output vector.
 * @return error code, 0 if successful.
 */
int pacmensl::StationaryMCSolver::ModifiedMatrixAction(Mat A, Vec x, Vec y) {
  int ierr;
  TIMatvec* mv;
  ierr = MatShellGetContext(A, &mv);
  CHKERRQ(ierr);
  ierr = (*mv)(x, y);
  PACMENSLCHKERRQ(ierr);

  return 0;
}

int pacmensl::StationaryMCSolver::Clear() {
  int ierr;
  if (ksp_ != nullptr){
    ierr = KSPDestroy(&ksp_);
    CHKERRQ(ierr);
  }
  mat_diagonal_ = nullptr;
  solution_ = nullptr;
  matvec_ = nullptr;
  if (inf_generator_ != nullptr){
    ierr = MatDestroy(&inf_generator_);
    CHKERRQ(ierr);
  }
  return 0;
}

pacmensl::StationaryMCSolver::~StationaryMCSolver() {
  PACMENSLCHKERREXCEPT(Clear());
  MPI_Comm_free(&comm_);
}

int pacmensl::StationaryMCSolver::Solve() {
  int ierr;
  ierr = KSPSolve(ksp_, *solution_, *mat_diagonal_);
  CHKERRQ(ierr);
  return 0;
}
