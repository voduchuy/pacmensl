//
// Created by Huy Vo on 12/6/18.
//

#ifndef PACMENSL_TSFSP_H
#define PACMENSL_TSFSP_H

#include "OdeSolverBase.h"
#include "PetscWrap.h"
#include "Sys.h"


namespace pacmensl {
class TsFsp: public OdeSolverBase {
 public:
  explicit TsFsp(MPI_Comm _comm);

  PacmenslErrorCode SetUp() override ;

  PetscInt Solve() override;

  int FreeWorkspace() override;

  ~TsFsp() override;

 protected:
  TSType type_ = TSRK;
  Petsc<TS> ts_;
  PetscReal t_now_tmp = 0.0;

  Vec solution_tmp_;

  static int TSRhsFunc(TS ts, PetscReal t, Vec u, Vec F, void* ctx);
  static int TSDetectFspError(TS ts,PetscReal t,Vec U,PetscScalar fvalue[],void* ctx);
};
}

#endif //PACMENSL_TSFSP_H
