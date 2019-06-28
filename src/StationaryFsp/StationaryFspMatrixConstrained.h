//
// Created by Huy Vo on 2019-06-24.
//

#ifndef PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPMATRIXCONSTRAINED_H_
#define PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPMATRIXCONSTRAINED_H_

#include "FspMatrixConstrained.h"
namespace pacmensl{
  class StationaryFspMatrixConstrained: public FspMatrixBase {
   public:
    explicit StationaryFspMatrixConstrained(MPI_Comm comm);
    int GenerateValues( const StateSetBase &fsp, const arma::Mat< int > &SM, const PropFun &prop,
                        void *prop_args, const TcoefFun &new_t_fun, void *t_fun_args) override;
    int Action(PetscReal t, Vec x, Vec y) override;
    int EvaluateOutflows(Vec sfsp_solution, arma::Row<PetscReal> &sinks);
    int Destroy() override;
    ~StationaryFspMatrixConstrained();
   protected:
    Vec diagonal_;
    int num_constraints_;
    std::vector<Mat> sinks_mat_; ///< local matrix to evaluate sink states
    Vec sink_entries_, sink_tmp;
  };
}
#endif //PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPMATRIXCONSTRAINED_H_
