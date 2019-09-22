//
// Created by Huy Vo on 12/6/18.
//

#ifndef PACMENSL_CVODEFSP_H
#define PACMENSL_CVODEFSP_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include<nvector/nvector_petsc.h>
#include "OdeSolverBase.h"
#include "StateSetConstrained.h"
#include "Sys.h"


namespace pacmensl {
class CvodeFsp : public OdeSolverBase {
 public:
  explicit CvodeFsp(MPI_Comm _comm, int lmm = CV_BDF);

  PacmenslErrorCode SetUp() override ;

  PetscInt Solve() override;

  int FreeWorkspace() override;

  ~CvodeFsp();
 protected:
  int lmm_ = CV_BDF;
  void *cvode_mem = nullptr;
  SUNLinearSolver linear_solver = nullptr;
  N_Vector solution_wrapper = nullptr;
  N_Vector solution_tmp = nullptr;
  N_Vector constr_vec_ = nullptr;

  PetscReal t_now_tmp = 0.0;
  int cvode_stat = 0;
  static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *solver);
  static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                       N_Vector u, N_Vector fu,
                       void *FPS_ptr, N_Vector tmp);
};
}

#endif //PACMENSL_CVODEFSP_H
