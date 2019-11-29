//
// Created by Huy Vo on 2019-06-24.
//

#ifndef PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPMATRIXCONSTRAINED_H_
#define PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPMATRIXCONSTRAINED_H_

#include "FspMatrixConstrained.h"
namespace pacmensl {
class StationaryFspMatrixConstrained : public FspMatrixBase
{
 public:
  explicit StationaryFspMatrixConstrained(MPI_Comm comm);
  PacmenslErrorCode GenerateValues(const StateSetBase &fsp,
                                   const arma::Mat<Int> &SM,
                                   std::vector<int> time_vayring,
                                   const TcoefFun &new_t_fun,
                                   const PropFun &prop,
                                   const std::vector<int> &enable_reactions,
                                   void *t_fun_args,
                                   void *prop_args) override;
  int Action(PetscReal t, Vec x, Vec y) override;
  int EvaluateOutflows(Vec sfsp_solution, arma::Row<PetscReal> &sinks);
  int Destroy() override;
  ~StationaryFspMatrixConstrained() override;
  friend class StationaryMCSolver;
  friend class StationaryFspSolverMultiSinks;
 protected:
  Vec              diagonal_;
  int              num_constraints_;
  std::vector<Petsc<Mat>> sinks_mat_; ///< local matrix to evaluate sink states

  Vec              sink_entries_, sink_tmp;
};
}
#endif //PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPMATRIXCONSTRAINED_H_
