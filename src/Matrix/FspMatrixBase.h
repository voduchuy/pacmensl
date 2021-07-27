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

#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <armadillo>
#include <mpi.h>
#include <petscmat.h>
#include <petscis.h>
#include "Model.h"
#include "Sys.h"
#include "StateSetBase.h"
#include "StateSetConstrained.h"
#include "PetscWrap.h"

namespace pacmensl {
using Real = PetscReal;
using Int = PetscInt;

/**
 * @brief Base class for the time-dependent FSP-truncated CME matrix.
 * @details We currently assume that the CME matrix could be decomposed into the form
 *  \f$ A(t) = \sum_{r=1}^{M}{c_r(t, \theta)A_r(\theta)} \f$
 * where c_r(t,\theta) are scalar-valued functions that depend on the time variable and parameters, while the matrices \f$ A_r(\theta) \f$ are time-independent.
 **/
class FspMatrixBase {
 public:
  /* constructors */
  /**
   * @brief Constructor for the base FSP matrix object.
   * @param comm Communicator context for the processes that own the matrix.
   */
  explicit FspMatrixBase(MPI_Comm comm);
  
  /**
   * @brief Populate the fields of the FSP matrix object.
   * @details __Collective__
   * @param fsp (in) The FSP-truncated state space from which the transition rate matrix is built.
   * @param SM (in) The stoichiometry matrix of the reaction network.
   * @param time_varying (in) List of reactions whose propensities are time-varying.
   * @param new_prop_t (in) Function pointer to evaluate the vector of time-varying coefficients $c_r(t,\theta)$ at time $t$.
   * @param new_prop_x (in) Function pointer to evaluate the time-independent component of the propensities.
   * @param enable_reactions (in) Vector of reactions that are active. If left empty, we assume all reactions are active. Propensity functions of inactive reactions are not added to the data structure.
   * @param prop_t_args (in) Pointer to additional data for the evaluation of time-varying coefficients.
   * @param prop_x_args (in) Pointer to additional data for the evaluation of the time-independent components of the propensity functions.
   * @return Error code. 0 if success, -1 otherwise.
   */
  virtual PacmenslErrorCode
  GenerateValues(const StateSetBase &fsp,
                 const arma::Mat<Int> &SM,
                 std::vector<int> time_varying,
                 const TcoefFun &new_prop_t,
                 const PropFun &new_prop_x,
                 const std::vector<int> &enable_reactions,
                 void *prop_t_args,
                 void *prop_x_args);
  
  PacmenslErrorCode SetTimeFun(TcoefFun new_t_fun, void *new_t_fun_args);
  
  virtual int Destroy();
  
  /**
  * @brief Compute y = A*x.
  * @details __Collective__
  * @param t (in) time.
  * @param x (in) input vector.
  * @param y (out) output vector.
  * @return error code, 0 if successful.
  */
  virtual PacmenslErrorCode Action(PetscReal t, Vec x, Vec y);
  
  /**
   * @brief Generate a PETSc Matrix object with the same sparsity structure as the FSP matrix. This serves as the Jacobian matrix required by ODEs solvers.
   * @details __Collective__
   * @param A (out) pointer to PETSc Mat object.
   * @return Error code. 0 if successful, -1 otherwise.
   */
  virtual PacmenslErrorCode CreateRHSJacobian(Mat *A);
  
  /**
   * @brief Populate the values of a PETSc Mat object with the entries of the FSP matrix at any given time.
   * @param t (in) time
   * @param A (out) PETSc Mat object to write the entries to.
   * @return Error code: 0 if successful, -1 otherwise.
   * @details __Collective__.
   *    The sparsity structure of A must be generated first with \ref CreateRHSJacobian().
   */
  virtual PacmenslErrorCode ComputeRHSJacobian(PetscReal t, Mat A);
  
  /**
   * @brief Get an estimate for the number of floating-point operations needed by the calling process per matrix-vector multiplication (i.e., per call to the \ref Action() method).
   * @param nflops (out) number of flops.
   * @return Error code: 0 if successful, -1 otherwise.
   * @details __Local__
   */
  virtual PacmenslErrorCode GetLocalMVFlops(PetscInt *nflops);
  
  /**
   * @brief Get the number of rows owned by the calling process.
   * @return Number of rows owned by the calling process.
   * @details __Local__
   */
  int GetNumLocalRows() const { return num_rows_local_; };
  
  /**
   * @brief Object destructor.
   */
  virtual ~FspMatrixBase();
 protected:
  MPI_Comm comm_ = MPI_COMM_NULL; ///< Communicator context
  int rank_; ///< Rank of local process
  int comm_size_; ///< Total number of processes that own this object
  
  Int num_reactions_ = 0; ///< Number of reactions
  Int num_rows_global_ = 0; ///< Global number of rows
  Int num_rows_local_ = 0; ///< Number of matrix rows owned by this process
  
  std::vector<int> enable_reactions_ = std::vector<int>();
  std::vector<int> tv_reactions_ = std::vector<int>();
  std::vector<int> ti_reactions_ = std::vector<int>();
  
  // Local data of the matrix
  std::vector<Petsc<Mat>> tv_mats_; ///< List of PETSc Mat objects for storing the $A_r$ factors for reactions $r$ with time-varying propensities
  Petsc<Mat> ti_mat_; ///< PETsc Mat object to store the time-independent term of the matrix sum decomposition
  
  // Data for computing the matrix action
  Petsc<Vec> work_; ///< Work vector for computing operator times vector
  
  TcoefFun t_fun_ = nullptr;  ///< Pointer to external function that evaluates time-varying factors of reaction propensities.
  void *t_fun_args_ = nullptr; ///< Pointer for extra data needed in \ref t_fun_ evaluation.
  arma::Row<Real> time_coefficients_; ///< Vector to store the time-varying factors of reaction propensities
  
  /**
   * @brief Populate the data members with information about the inter-process layout of the object.
   * @param fsp (in) FSP-truncated state space from which the matrix is based.
   * @return Error code: 0 if successful, -1 otherwise.
   * @details __Collective__
   * Modified fields: \ref num_rows_local_, \ref num_rows_global_, \ref work_.
   */
  virtual int DetermineLayout_(const StateSetBase &fsp);
  
  // arrays for counting nonzero entries on the diagonal and off-diagonal blocks
  arma::Mat<Int> dblock_nz_, oblock_nz_;
  arma::Col<Int> ti_dblock_nz_, ti_oblock_nz_;
  // arrays of nonzero column indices
  arma::Mat<Int> offdiag_col_idxs_;
  // array of matrix values
  arma::Mat<PetscReal> offdiag_vals_;
  arma::Mat<PetscReal> diag_vals_;
};

}
