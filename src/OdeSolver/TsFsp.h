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

  PacmenslErrorCode SetTsType(std::string type);

  int FreeWorkspace() override;

  ~TsFsp() override;

 protected:
  std::string type_ = std::string(TSROSW);
  Petsc<TS> ts_;
  PetscReal t_now_tmp = 0.0;
  PetscInt fsp_stop_ = 0;
  Vec solution_tmp_;

  int njac = 0, nstep = 0;

  Mat J = nullptr;
  static int TSRhsFunc(TS ts, PetscReal t, Vec u, Vec F, void* ctx);
  static int TSJacFunc(TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx);
  static int TSIFunc(TS ts, PetscReal t, Vec u, Vec u_t, Vec F, void*ctx);
  static int TSIJacFunc(TS ts, PetscReal t, Vec u, Vec u_t, PetscReal a, Mat A, Mat P, void*ctx);
  static int TSCheckFspError(TS ts);

  static int CheckImplicitType(TSType type, int* implicit);
};
}

#endif //PACMENSL_TSFSP_H
