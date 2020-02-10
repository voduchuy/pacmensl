//
// Created by Huy Vo on 2019-06-27.
//

#ifndef PACMENSL_SRC_SENSFSP_SENSFSPMATRIX_H_
#define PACMENSL_SRC_SENSFSP_SENSFSPMATRIX_H_

#include "SensModel.h"
#include "FspMatrixBase.h"
#include "FspMatrixConstrained.h"

namespace pacmensl {

/**
 * @brief Templated class for the time-dependent matrix on the right hand side of the finite state projection-based forward sensitivity
 * equation.
 * @details We currently assume that the CME matrix could be decomposed into the form
 *  \f$ A(t) = \sum_{r=1}^{M}{c_r(t, \theta)A_r} \f$
 * where c_r(t,\theta) are scalar-valued functions that depend on the time variable and parameters, while the matrices \f$ A_r \f$ are constant.
 * As a consequence, the derivative of \f$ A(t) \f$ with respect to the \f$ j \f$-th parameter component is assumed to have the form
 * \f$ \partial_{j}A(t) = \sum_{r=1}^{M}{\partial_j{c_r}(t,\theta)}A_r \f$
 */
template<typename FspMatrixT>
class SensFspMatrix
{
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
  MPI_Comm comm_ = nullptr;
  int      rank_ = 0;

  int                     num_parameters_ = 0; ///< Number of sensitivity parameters
  FspMatrixT              A_; ///< The FSP-truncated infinitesimal generator
  std::vector<FspMatrixT> dA_; ///< The derivatives of A with respect to sensitivity parameters
};
}

template<typename FspMatrixT>
pacmensl::SensFspMatrix<FspMatrixT>::SensFspMatrix(MPI_Comm comm) : A_(comm)
{
  int ierr;
  if (comm == MPI_COMM_NULL)
  {
    std::cout << "Null pointer detected.\n";
  }
  comm_ = comm;
  ierr = MPI_Comm_rank(comm, &rank_);
  PACMENSLCHKERRTHROW(ierr);
}

template<typename FspMatrixT>
PacmenslErrorCode pacmensl::SensFspMatrix<FspMatrixT>::Destroy()
{
  PacmenslErrorCode ierr;
  ierr = A_.Destroy();
  PACMENSLCHKERRQ(ierr);
  for (int i{0}; i < num_parameters_; ++i)
  {
    ierr = dA_[i].Destroy();
    PACMENSLCHKERRQ(ierr);
  }
  dA_.clear();
  return 0;
}

template<typename FspMatrixT>
pacmensl::SensFspMatrix<FspMatrixT>::~SensFspMatrix()
{
  Destroy();
  comm_ = nullptr;
}

template<typename FspMatrixT>
PacmenslErrorCode pacmensl::SensFspMatrix<FspMatrixT>::GenerateValues(const pacmensl::StateSetBase &state_set,
                                                                      const pacmensl::SensModel &model)
{
  PacmenslErrorCode ierr;
  ierr = A_.GenerateValues(state_set, model.stoichiometry_matrix_, model.tv_reactions_, model.prop_t_, model.prop_x_,
                           std::vector<int>(), model.prop_t_args_, model.prop_x_args_);
  PACMENSLCHKERRQ(ierr);
  num_parameters_ = model.num_parameters_;
  dA_.reserve(num_parameters_);
  if (!model.dpropensity_ic_.empty())
  {
    for (int i{0}; i < num_parameters_; ++i)
    {
      auto             first = model.dpropensity_ic_.begin() + model.dpropensity_rowptr_[i];
      auto             last  = model.dpropensity_ic_.begin() + model.dpropensity_rowptr_[i + 1];
      std::vector<int> enable_reactions(first, last);
      dA_.emplace_back(FspMatrixT(comm_));
      ierr                   = dA_[i].GenerateValues(state_set,
                                                     model.stoichiometry_matrix_,
                                                     model.tv_reactions_,
                                                     model.dprop_t_[i],
                                                     model.dprop_x_[i],
                                                     enable_reactions,
                                                     model.dprop_t_args_[i],
                                                     model.dprop_x_args_[i]);
      PACMENSLCHKERRQ(ierr);
    }
  } else
  {
    for (int i{0}; i < num_parameters_; ++i)
    {
      dA_.emplace_back(FspMatrixT(comm_));
      ierr = dA_[i].GenerateValues(state_set,
                                   model.stoichiometry_matrix_,
                                   model.tv_reactions_,
                                   model.dprop_t_[i],
                                   model.dprop_x_[i],
                                   std::vector<int>(),
                                   model.dprop_t_args_[i],
                                   model.dprop_x_args_[i]);
      PACMENSLCHKERRQ(ierr);
    }
  }
  return 0;
}

template<typename FspMatrixT>
PacmenslErrorCode pacmensl::SensFspMatrix<FspMatrixT>::Action(PetscReal t, Vec x, Vec y)
{
  return A_.Action(t, x, y);
}

template<typename FspMatrixT>
PacmenslErrorCode pacmensl::SensFspMatrix<FspMatrixT>::SensAction(int i_par, PetscReal t, Vec x, Vec y)
{
  if (i_par < 0 || i_par >= num_parameters_)
  {
    return -1;
  }
  return dA_[i_par].Action(t, x, y);
}

template<typename FspMatrixT>
int pacmensl::SensFspMatrix<FspMatrixT>::GetNumLocalRows() const
{
  return A_.GetNumLocalRows();
}

#endif //PACMENSL_SRC_SENSFSP_SENSFSPMATRIX_H_
