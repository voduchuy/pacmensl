//
// Created by Huy Vo on 12/6/18.
//

#ifndef PECMEAL_CVODEFSP_H
#define PECMEAL_CVODEFSP_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include<nvector/nvector_petsc.h>
#include "OdeSolverBase.h"
#include "StateSetConstrained.h"
#include "cme_util.h"

#ifndef NDEBUG
#define CVODECHKERR(comm, flag){\
    if (flag < 0) \
    {\
    PetscPrintf(comm, "\nSUNDIALS_ERROR: function failed in file %s line %d with flag = %d\n\n",\
    __FILE__,__LINE__, flag);\
    MPI_Abort(comm, 1);\
    }\
    }
#else
#define CVODECHKERR(comm_, flag){while(false){}};
#endif

namespace pecmeal {
class CvodeFsp : public OdeSolverBase {
 protected:
  void *cvode_mem = nullptr;
  SUNLinearSolver linear_solver = nullptr;
  N_Vector solution_wrapper = nullptr;
  PetscReal t_now_tmp = 0.0;
  PetscReal rel_tol = 1.0e-4;
  PetscReal abs_tol = 1.0e-8;
  int cvode_stat;
  static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *solver);
  static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                       N_Vector u, N_Vector fu,
                       void *FPS_ptr, N_Vector tmp);
  N_Vector solution_tmp = nullptr;
 public:
  explicit CvodeFsp(MPI_Comm _comm, int lmm = CV_BDF);
  void SetCVodeTolerances(PetscReal _r_tol, PetscReal _abs_tol);

  PetscInt solve() override;

  void free() override;
  ~CvodeFsp();
};
}

#endif //PECMEAL_CVODEFSP_H
