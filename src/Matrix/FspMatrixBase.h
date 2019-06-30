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
 *  \f$ A(t) = \sum_{r=1}^{M}{c_r(t, \theta)A_r} \f$
 * where c_r(t,\theta) are scalar-valued functions that depend on the time variable and parameters, while the matrices \f$ A_r \f$ are constant.
 **/
class FspMatrixBase {
 public:
  /* constructors */
  explicit FspMatrixBase(MPI_Comm comm);
  FspMatrixBase(const FspMatrixBase &A); // untested
  FspMatrixBase(FspMatrixBase &&A) noexcept; // untested

  /* Assignments */
  FspMatrixBase &operator=(const FspMatrixBase &A);
  FspMatrixBase &operator=(FspMatrixBase &&A) noexcept;

  virtual PacmenslErrorCode
  GenerateValues(const StateSetBase &fsp, const arma::Mat<Int> &SM, const PropFun &propensity, void *propensity_args,
                 const TcoefFun &new_t_fun, void *t_fun_args);

  virtual int Destroy();

  virtual int Action(PetscReal t, Vec x, Vec y);

  PetscInt GetLocalGhostLength() const;

  int GetNumLocalRows() const { return num_rows_local_; };

  virtual ~FspMatrixBase();
 protected:
  MPI_Comm comm_ = nullptr;
  int      rank_;
  int      comm_size_;

  Int num_reactions_   = 0;
  Int num_rows_global_ = 0;
  Int num_rows_local_  = 0;

  // Local data of the matrix
  std::vector<Petsc<Mat>> diag_mats_;
  std::vector<Petsc<Mat>> offdiag_mats_;

  // Data for computing the matrix action
  Vec        work_        = nullptr; ///< Work vector for computing operator times vector
  Vec        lvec_        = nullptr; ///< Local vector to receive scattered data from the input vec
  Vec        xx           = nullptr, yy = nullptr, zz = nullptr; ///< Local portion of the vectors
  PetscInt   lvec_length_ = 0; ///< Number of ghost entries owned by the local process
  VecScatter action_ctx_  = nullptr; ///< Scatter context for computing matrix action

  TcoefFun        t_fun_       = nullptr;
  void            *t_fun_args_ = nullptr;
  arma::Row<Real> time_coefficients_;

  virtual int DetermineLayout_(const StateSetBase &fsp);
};

}
