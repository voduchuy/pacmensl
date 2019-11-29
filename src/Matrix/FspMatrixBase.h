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

  virtual PacmenslErrorCode
  GenerateValues(const StateSetBase &fsp,
                 const arma::Mat<Int> &SM,
                 std::vector<int> time_vayring,
                 const TcoefFun &new_prop_t,
                 const PropFun &new_prop_x,
                 const std::vector<int> &enable_reactions,
                 void *prop_t_args,
                 void *prop_x_args);


  PacmenslErrorCode SetTimeFun(TcoefFun new_t_fun, void *new_t_fun_args);

  virtual int Destroy();

  virtual PacmenslErrorCode Action(PetscReal t, Vec x, Vec y);

  virtual int CreateRHSJacobian(Mat* A);
  virtual int ComputeRHSJacobian(PetscReal t, Mat A);

  int GetNumLocalRows() const { return num_rows_local_; };

  virtual ~FspMatrixBase();
 protected:
  MPI_Comm comm_ = nullptr;
  int      rank_;
  int      comm_size_;

  Int num_reactions_   = 0;
  Int num_rows_global_ = 0;
  Int num_rows_local_  = 0;
  std::vector<int> enable_reactions_;
  std::vector<int> tv_reactions_;
  std::vector<int> ti_reactions_;

  // Local data of the matrix
  std::vector<Petsc<Mat>> tv_mats_;
  Petsc<Mat> ti_mat_;

  // Data for computing the matrix action
  Petsc<Vec>        work_; ///< Work vector for computing operator times vector

  TcoefFun        t_fun_       = nullptr;
  void            *t_fun_args_ = nullptr;
  arma::Row<Real> time_coefficients_;

  virtual int DetermineLayout_(const StateSetBase &fsp);
};

}
