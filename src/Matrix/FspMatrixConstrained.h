//
// Created by Huy Vo on 6/2/19.
//

#ifndef PACMENSL_FSPMATRIXCONSTRAINED_H
#define PACMENSL_FSPMATRIXCONSTRAINED_H

#include "FspMatrixBase.h"

namespace pacmensl {
class FspMatrixConstrained : public FspMatrixBase {
 public:
  explicit FspMatrixConstrained(MPI_Comm comm);

  FspMatrixConstrained(const FspMatrixConstrained &A); // untested
  FspMatrixConstrained(FspMatrixConstrained &&A) noexcept; // untested

  /* Assignments */
  FspMatrixConstrained& operator=(const FspMatrixConstrained &A);
  FspMatrixConstrained& operator=(FspMatrixConstrained &&A) noexcept;

  PacmenslErrorCode GenerateValues(const StateSetBase &state_set,
                                   const arma::Mat<Int> &SM,
                                   const TcoefFun &new_prop_t,
                                   const PropFun &prop,
                                   const std::vector<int> &enable_reactions,
                                   void *prop_t_args,
                                   void *prop_args) override;

  int Destroy() override;

  int Action(PetscReal t, Vec x, Vec y) override;

  ~FspMatrixConstrained() override;

 protected:
  int              num_constraints_ = 0;
  int              sinks_rank_      = 0; ///< rank of the processor that stores sink states
  std::vector<int> sink_global_indices; ///< global indices of the sink states
  std::vector<Mat> local_sinks_mat_; ///< local matrices to evaluate sink states
  Vec              sink_entries_    = nullptr, sink_tmp = nullptr; ///< local vectors for computing local_sinks_mat[i]*x

  arma::Mat<int>                    sink_nnz; ///< sink_nnz(i, j) is the number of nonzeros of the i-th row of the j-th local sink matrix
  std::vector<std::vector<arma::Row<int>>>       sink_inz; ///< sink_inz[i][j] is a row containing local column indices of nonzero elements of the j-th row of the i-th local sink matrix
  std::vector<std::vector<arma::Row<PetscReal>>> sink_rows; ///< sink_rows[i][j] is the array of values coressponding to the indices in sink_inz

  bool parallel_sink_mats_generated = false;
  std::vector<Petsc<Mat>> parallel_sinks_mat_; ///< global sink matrices for forming the Jacobian
  PacmenslErrorCode GenerateParallelSinkMats();

  VecScatter sink_scatter_ctx_ = nullptr; ///< scatter context for collecting the global sink entries, these are located at the very end of the input vector during matrix-vector multiplication

  PacmenslErrorCode DetermineLayout_(const StateSetBase &fsp) override;

  int CreateRHSJacobianBasic(Mat* A) override;
  int ComputeRHSJacobianBasic(PetscReal t, Mat A) override;
  int CreateRHSJacobianCustom(Mat *A) override;
  int ComputeRHSJacoianCustom(Mat *A) override;
};
}

#endif //PACMENSL_FSPMATRIXCONSTRAINED_H
