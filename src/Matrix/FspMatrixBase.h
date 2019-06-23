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

namespace pacmensl {
using Real = PetscReal;
using Int = PetscInt;


/// Distributed data type for the truncated CME operator on a hyper-rectangle
/**
 *
 **/
class FspMatrixBase {

 protected:
  MPI_Comm comm_ = nullptr;
  int my_rank_;
  int comm_size_;

  Int num_reactions_;
  Int n_rows_global_;
  Int n_rows_local_;

  // Local data of the matrix
  std::vector<Mat> diag_mats_;
  std::vector<Mat> offdiag_mats_;

  // Data for computing the matrix action
  Vec work_; ///< Work vector for computing operator times vector
  Vec lvec_; ///< Local vector to receive scattered data from the input vec
  Vec xx, yy, zz; ///< Local portion of the vectors
  PetscInt lvec_length_; ///< Number of ghost entries owned by the local process
  VecScatter action_ctx_; ///< Scatter context for computing matrix action

  TcoefFun t_fun_ = nullptr;
  void* t_fun_args_ = nullptr;
  arma::Row< Real > time_coefficients_;

  virtual int DetermineLayout_( const StateSetBase &fsp);

 public:
  NOT_COPYABLE_NOT_MOVABLE(FspMatrixBase);

  /* constructors */
  explicit FspMatrixBase(MPI_Comm comm);

  virtual int
  GenerateValues( const StateSetBase &fsp, const arma::Mat< Int > &SM, const PropFun &propensity, void *propensity_args,
                  const TcoefFun &new_t_fun, void *t_fun_args );

  virtual int Destroy();

  virtual int action( PetscReal t, Vec x, Vec y);

  PetscInt get_local_ghost_length() const;

  int get_num_rows_local() const { return n_rows_local_; };

  ~FspMatrixBase();
};

}
