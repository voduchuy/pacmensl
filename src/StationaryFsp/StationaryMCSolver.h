//
// Created by Huy Vo on 2019-06-25.
//

#ifndef PACMENSL_SRC_STATIONARYFSP_STATIONARYMCSOLVER_H_
#define PACMENSL_SRC_STATIONARYFSP_STATIONARYMCSOLVER_H_

#include "StationaryFspMatrixConstrained.h"
namespace pacmensl
{
using TIMatvec=std::function<int(Vec x, Vec y)>;

class StationaryMCSolver {
 public:
  NOT_COPYABLE_NOT_MOVABLE(StationaryMCSolver);
  explicit StationaryMCSolver(MPI_Comm comm_);
  int SetSolutionVec(Vec* vec);
  int SetMatDiagonal(Vec* diag);
  int SetMatVec(const TIMatvec &matvec);
  int EnableStatusPrinting();
  int SetUp();
  int Solve();
  int Clear();
  ~StationaryMCSolver();
 protected:
  MPI_Comm comm_ = MPI_COMM_NULL;
  Vec* solution_ = nullptr;
  Vec* mat_diagonal_ = nullptr;
  TIMatvec matvec_ = nullptr;

  bool print_status_ = false;

  std::unique_ptr<Petsc<Mat>> inf_generator_;
  std::unique_ptr<Petsc<KSP>> ksp_;
  int n_local_, n_global_;
  static int ModifiedMatrixAction(Mat A, Vec x, Vec y);
};

}
#endif //PACMENSL_SRC_STATIONARYFSP_STATIONARYMCSOLVER_H_
