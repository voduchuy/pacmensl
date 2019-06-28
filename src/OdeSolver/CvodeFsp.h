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


#define CVODECHKERRABORT(comm, flag){\
    if (flag < 0) \
    {\
    PetscPrintf(comm, "\nSUNDIALS_ERROR: function failed in file %s line %d with flag = %d\n\n",\
    __FILE__,__LINE__, flag);\
    MPI_Abort(comm, 1);\
    }\
    }
#define CVODECHKERRQ(flag){\
    if (flag < 0) \
    {\
    int rank;\
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);\
    printf("\nSUNDIALS_ERROR: function failed on rank %d in file %s line %d with flag = %d\n\n",\
    rank,__FILE__,__LINE__, flag);\
    return -1;\
    }\
    }


namespace pacmensl {
class CvodeFsp : public OdeSolverBase {
 protected:
  int lmm_;
  void *cvode_mem = nullptr;
  SUNLinearSolver linear_solver = nullptr;
  N_Vector solution_wrapper = nullptr;
  PetscReal t_now_tmp = 0.0;
  PetscReal rel_tol = 1.0e-6;
  PetscReal abs_tol = 1.0e-14;
  int cvode_stat = 0;
  static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *solver);
  static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                       N_Vector u, N_Vector fu,
                       void *FPS_ptr, N_Vector tmp);
  N_Vector solution_tmp = nullptr;
 public:
  explicit CvodeFsp(MPI_Comm _comm, int lmm = CV_BDF);
  int SetCVodeTolerances(PetscReal _r_tol, PetscReal _abs_tol);

  PetscInt Solve() override;

  int FreeWorkspace() override;

  ~CvodeFsp();
};
}

#endif //PACMENSL_CVODEFSP_H
