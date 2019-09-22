//
// Created by Huy Vo on 2019-06-25.
//

#include "StationaryMCSolver.h"

pacmensl::StationaryMCSolver::StationaryMCSolver(MPI_Comm comm) {
  int ierr;
  comm_ = comm;
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
  ierr = VecGetLocalSize(*solution_, &n_local_); CHKERRQ(ierr);
  ierr = VecGetSize(*solution_, &n_global_); CHKERRQ(ierr);

  inf_generator_ = std::unique_ptr<Petsc<Mat>>(new Petsc<Mat>);
  ierr = MatCreateShell(comm_, n_local_, n_local_, n_global_, n_global_, ( void * ) this,
                        inf_generator_->mem()); CHKERRQ(ierr);
  ierr = MatSetOperation(*inf_generator_, MATOP_MULT, ( void (*)()) &ModifiedMatrixAction); CHKERRQ(ierr);
  // Create Krylov solver object
  ksp_ = std::unique_ptr<Petsc<KSP>>(new Petsc<KSP>);
  ierr = KSPCreate(comm_, ksp_->mem()); CHKERRQ(ierr);
  ierr = KSPSetOperators(*ksp_, *inf_generator_, *inf_generator_); CHKERRQ(ierr);
  ierr = KSPSetType(*ksp_, KSPGMRES); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(*ksp_, PETSC_TRUE); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(*ksp_); CHKERRQ(ierr);
  ierr = KSPSetUp(*ksp_); CHKERRQ(ierr);
  return 0;
}

int pacmensl::StationaryMCSolver::SetMatVec(const TIMatvec &matvec) {
  matvec_ = matvec;
  return 0;
}

/**
 * @brief Given the infinitesimal generator A of the discrete Markov process, compute the action
 * \f$ \left(A + \frac{2}{n} d q^T \right)v \f$
 * where n is the number of rows of A, d = diag(A), q = (1,1,..,1)^T.
 * @details Collective.
 * @param A infinitesimal generator of the Markov chain.
 * @param x input vector.
 * @param y output vector.
 * @return error code, 0 if successful.
 */
int pacmensl::StationaryMCSolver::ModifiedMatrixAction(Mat A, Vec x, Vec y) {
  int ierr;
  StationaryMCSolver* ctx;
  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);
  ierr = (ctx->matvec_)(x, y); PACMENSLCHKERRQ(ierr);
  PetscReal alpha;
  ierr = VecSum(x, &alpha); CHKERRQ(ierr);
  alpha = alpha*2.0/ctx->n_global_;
  ierr = VecAXPY(y, alpha, *ctx->mat_diagonal_); CHKERRQ(ierr);
  return 0;
}

int pacmensl::StationaryMCSolver::Clear() {
  int ierr;
  mat_diagonal_ = nullptr;
  solution_ = nullptr;
  matvec_ = nullptr;
  return 0;
}

pacmensl::StationaryMCSolver::~StationaryMCSolver() {
  Clear();
  comm_ = nullptr;
}

int pacmensl::StationaryMCSolver::Solve() {
  int ierr;
  PetscReal alpha;
  ierr = KSPSolve(*ksp_, *mat_diagonal_, *solution_); CHKERRQ(ierr);
  ierr = VecSum(*solution_, &alpha); CHKERRQ(ierr);
  ierr = VecScale(*solution_, 1.0/alpha); CHKERRQ(ierr);
  return 0;
}






