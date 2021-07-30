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

#ifndef PACMENSL_SRC_SENSFSP_SENSFSPMATRIX_H_
#define PACMENSL_SRC_SENSFSP_SENSFSPMATRIX_H_

#include "SensModel.h"
#include "FspMatrixBase.h"
#include "FspMatrixConstrained.h"
#include "PetscWrap.h"

namespace pacmensl {

/**
 * @brief Templated class for the time-dependent matrix on the right hand side of the finite state projection-based forward sensitivity
 * equation.
 * @details We currently assume that the CME matrix could be decomposed into the form
 *  \f$ A(t) = \sum_{r=1}^{M}{c_r(t, \theta)A_r(\theta)} \f$
 * where c_r(t,\theta) are scalar-valued functions that depend on the time variable and parameters, while the matrices \f$ A_r(\theta) \f$ depend only on parameters.
 * As a consequence, the derivative of \f$ A(t) \f$ with respect to the \f$ j \f$-th parameter component is assumed to have the form
 * \f$ \partial_{j}A(t) = \sum_{r=1}^{M}{\partial_j{c_r}(t,\theta)}A_r(\theta) \f$
 */
template<typename FspMatrixT>
class SensFspMatrix {
 public:
  NOT_COPYABLE_NOT_MOVABLE(SensFspMatrix);

  explicit SensFspMatrix(MPI_Comm comm);
  virtual PacmenslErrorCode GenerateValues(const StateSetBase &state_set, const SensModel &model);
  virtual PacmenslErrorCode Action(PetscReal t, Vec x, Vec y);
  virtual PacmenslErrorCode SensAction(int i_par, PetscReal t, Vec x, Vec y);
  virtual PacmenslErrorCode Destroy();
  virtual ~SensFspMatrix();

  int GetNumLocalRows() const;
 protected:
  MPI_Comm comm_ = MPI_COMM_NULL;
  int rank_ = 0;

  int num_parameters_ = 0; ///< Number of sensitivity parameters
  FspMatrixT A_; ///< The FSP-truncated infinitesimal generator
  std::vector<FspMatrixT> dcxA_; ///< The derivatives of A with respect to sensitivity parameters
  std::vector<FspMatrixT> cxdA_;
  Petsc<Vec> work_; ///< Scratch vector for performing matrix-vector multiplication
};
}

template<typename FspMatrixT>
pacmensl::SensFspMatrix<FspMatrixT>::SensFspMatrix(MPI_Comm comm) : A_(comm) {
  int ierr;
  if (comm == MPI_COMM_NULL) {
    std::cout << "Null pointer detected.\n";
  }
  comm_ = comm;
  ierr = MPI_Comm_rank(comm, &rank_);
  PACMENSLCHKERRTHROW(ierr);
}

template<typename FspMatrixT>
PacmenslErrorCode pacmensl::SensFspMatrix<FspMatrixT>::Destroy() {
  PacmenslErrorCode ierr;
  ierr = A_.Destroy();
  PACMENSLCHKERRQ(ierr);
  for (int i{0}; i < num_parameters_; ++i) {
    ierr = dcxA_[i].Destroy();
    PACMENSLCHKERRQ(ierr);
    ierr = cxdA_[i].Destroy();
    PACMENSLCHKERRQ(ierr);
  }
  dcxA_.clear();
  cxdA_.clear();
  return 0;
}

template<typename FspMatrixT>
pacmensl::SensFspMatrix<FspMatrixT>::~SensFspMatrix() {
  Destroy();
  comm_ = MPI_COMM_NULL;
}

template<typename FspMatrixT>
PacmenslErrorCode
pacmensl::SensFspMatrix<FspMatrixT>::GenerateValues(const pacmensl::StateSetBase &state_set,
                                                    const pacmensl::SensModel &model) {
  PacmenslErrorCode ierr;
  ierr = A_.GenerateValues(state_set,
                           model.stoichiometry_matrix_,
                           model.tv_reactions_,
                           model.prop_t_,
                           model.prop_x_,
                           std::vector<int>(),
                           model.prop_t_args_,
                           model.prop_x_args_);
  PACMENSLCHKERRQ(ierr);

  ierr = VecCreate(comm_, work_.mem());
  CHKERRQ(ierr);
  ierr = VecSetSizes(work_, A_.GetNumLocalRows(), PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(work_);
  CHKERRQ(ierr);

  num_parameters_ = model.num_parameters_;

  // Generate the first part of the derivative of A wrt parameters, consisting of the terms dc*A
  dcxA_.reserve(num_parameters_);
  if (!model.dprop_t_sp_.empty()) {
    for (int i{0}; i < num_parameters_; ++i) {
      dcxA_.emplace_back(FspMatrixT(comm_));
      if (!model.dprop_t_sp_[i].empty()) {
        auto dprop_t = std::bind(model.dprop_t_, i,
                                 std::placeholders::_1,
                                 std::placeholders::_2,
                                 std::placeholders::_3,
                                 std::placeholders::_4);
        ierr = dcxA_[i].GenerateValues(state_set,
                                       model.stoichiometry_matrix_,
                                       model.tv_reactions_,
                                       dprop_t,
                                       model.prop_x_,
                                       model.dprop_t_sp_[i],
                                       model.dprop_t_args_,
                                       model.prop_x_args_);
        PACMENSLCHKERRQ(ierr);
      }
    }
  } else {
    for (int i{0}; i < num_parameters_; ++i) {
      dcxA_.emplace_back(FspMatrixT(comm_));
    }
  }

  // Generate the second part of the derivative of A wrt parameters, consisting of the terms c*dA
  cxdA_.reserve(num_parameters_);
  if (!model.dprop_x_sp_.empty()) {
    for (int i{0}; i < num_parameters_; ++i) {
      cxdA_.emplace_back(FspMatrixT(comm_));
      if (!model.dprop_x_sp_[i].empty()) {
        auto dprop_x = std::bind(model.dprop_x_, i,
                                 std::placeholders::_1,
                                 std::placeholders::_2,
                                 std::placeholders::_3,
                                 std::placeholders::_4,
                                 std::placeholders::_5,
                                 std::placeholders::_6);

        ierr = cxdA_[i].GenerateValues(state_set,
                                       model.stoichiometry_matrix_,
                                       model.tv_reactions_,
                                       model.prop_t_,
                                       dprop_x,
                                       model.dprop_x_sp_[i],
                                       model.prop_t_args_,
                                       model.dprop_x_args_);
        PACMENSLCHKERRQ(ierr);
      }
    }
  } else {
    for (int i{0}; i < num_parameters_; ++i) {
      cxdA_.emplace_back(FspMatrixT(comm_));
    }
  }
  return 0;
}

template<typename FspMatrixT>
PacmenslErrorCode
pacmensl::SensFspMatrix<FspMatrixT>::Action(PetscReal t, Vec x, Vec y) {
  return A_.Action(t, x, y);
}

template<typename FspMatrixT>
PacmenslErrorCode
pacmensl::SensFspMatrix<FspMatrixT>::SensAction(int i_par, PetscReal t, Vec x, Vec y) {
  int ierr;
  if (i_par < 0 || i_par >= num_parameters_) {
    return -1;
  }
  ierr = dcxA_[i_par].Action(t, x, y);
  PACMENSLCHKERRQ(ierr);
  ierr = cxdA_[i_par].Action(t, x, work_);
  PACMENSLCHKERRQ(ierr);
  ierr = VecAXPY(y, 1.0, work_);
  CHKERRQ(ierr);
  return 0;
}

template<typename FspMatrixT>
int pacmensl::SensFspMatrix<FspMatrixT>::GetNumLocalRows() const {
  return A_.GetNumLocalRows();
}

#endif //PACMENSL_SRC_SENSFSP_SENSFSPMATRIX_H_
