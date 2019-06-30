//
// Created by Huy Vo on 2019-06-28.
//

#ifndef PACMENSL_SRC_SENSFSP_FORWARDSENSCVODEFSP_H_
#define PACMENSL_SRC_SENSFSP_FORWARDSENSCVODEFSP_H_

#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<nvector/nvector_petsc.h>
#include<cvodes/cvodes.h>
#include <cvodes/cvodes_spils.h>
#include "ForwardSensSolverBase.h"


namespace pacmensl {
class ForwardSensCvodeFsp : public ForwardSensSolverBase {
 public:

  explicit ForwardSensCvodeFsp(MPI_Comm comm) : ForwardSensSolverBase(comm) {};

  PacmenslErrorCode SetUp() override;

  PetscInt Solve() override; // Advance the solution_ toward final time. Return 0 if reaching final time, 1 if the Fsp criteria fails before reaching final time, -1 if fatal errors.

  PacmenslErrorCode FreeWorkspace() override;

  ~ForwardSensCvodeFsp() override;
 protected:
  int             lmm_          = CV_BDF;
  void            *cvode_mem    = nullptr;
  SUNLinearSolver linear_solver = nullptr;

  N_Vector              solution_wrapper = nullptr;
  std::vector<N_Vector> sens_vecs_wrapper;

  N_Vector solution_tmp = nullptr;
  N_Vector *sens_vecs_tmp = nullptr;

  PetscReal t_now_tmp_ = 0.0;
  PetscReal rel_tol    = 1.0e-4;
  PetscReal abs_tol    = 1.0e-8;
  int       cvode_stat = 0;
  static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *data);
  static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                       N_Vector u, N_Vector fu,
                       void *FPS_ptr, N_Vector tmp);
  static int cvsens_rhs(int Ns, PetscReal t, N_Vector y, N_Vector ydot, int iS, N_Vector yS, N_Vector ySdot,
                        void *user_data, N_Vector tmp1, N_Vector tmp2);
};
}

#endif //PACMENSL_SRC_SENSFSP_FORWARDSENSCVODEFSP_H_
