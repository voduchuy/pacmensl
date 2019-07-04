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
  std::vector<Mat> sinks_mat_; ///< local matrix to evaluate sink states
  Vec              sink_entries_    = nullptr, sink_tmp = nullptr;

  VecScatter sink_scatter_ctx_ = nullptr;

  PacmenslErrorCode DetermineLayout_(const StateSetBase &fsp) override;
};
}

#endif //PACMENSL_FSPMATRIXCONSTRAINED_H
