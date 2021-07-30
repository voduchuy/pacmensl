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

#ifndef PACMENSL_FSPMATRIXCONSTRAINED_H
#define PACMENSL_FSPMATRIXCONSTRAINED_H

#include "FspMatrixBase.h"

namespace pacmensl {

/**
 * @brief FSP matrix corresponding to the FSP variant defined by inequality constraints.
 */
class FspMatrixConstrained : public FspMatrixBase
{
 public:
  explicit FspMatrixConstrained(MPI_Comm comm);
  
  PacmenslErrorCode
  GenerateValues(const StateSetBase &fsp,
                 const Model &model) override;
  
  PacmenslErrorCode GenerateValues(const StateSetBase &state_set,
                                   const arma::Mat<Int> &SM,
                                   std::vector<int> time_vayring,
                                   const TcoefFun &new_prop_t,
                                   const PropFun &prop,
                                   const std::vector<int> &enable_reactions,
                                   void *prop_t_args,
                                   void *prop_args) override;

  int Destroy() override;
  
  int Action(PetscReal t, Vec x, Vec y) override;

  int CreateRHSJacobian(Mat* A) override;
  int ComputeRHSJacobian(PetscReal t,Mat A) override;

  PacmenslErrorCode GetLocalMVFlops(PetscInt* nflops) override;

  ~FspMatrixConstrained() override;

 protected:
  int              num_constraints_ = 0;
  int              sinks_rank_      = 0; ///< rank of the processor that stores sink states
  std::vector<Mat> tv_sinks_mat_; ///< local matrix to evaluate sink states
  Mat              ti_sinks_mat_ = nullptr;
  Vec              sink_entries_    = nullptr, sink_tmp = nullptr;

  VecScatter sink_scatter_ctx_ = nullptr;

  Vec xx = nullptr;

  PacmenslErrorCode DetermineLayout_(const StateSetBase &fsp) override;

  arma::Mat<int>                    sinkmat_nnz;
  std::vector<arma::Row<int>>       sinkmat_inz;
  std::vector<arma::Row<PetscReal>> sinkmat_entries;
};
}

#endif //PACMENSL_FSPMATRIXCONSTRAINED_H
