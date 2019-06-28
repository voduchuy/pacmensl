//
// Created by Huy Vo on 2019-06-12.
//

#ifndef PACMENSL_SRC_ODESOLVER_KRYLOVFSP_H_
#define PACMENSL_SRC_ODESOLVER_KRYLOVFSP_H_

#include "OdeSolverBase.h"

namespace pacmensl {
class KrylovFsp : public OdeSolverBase {
 public:

  explicit KrylovFsp(MPI_Comm comm);

  virtual int SetUp() override;

  PetscInt Solve() override;

  int FreeWorkspace() override;

  ~KrylovFsp();

 protected:

  const int max_reject_ = 10000;
  PetscReal delta_ = 1.2, gamma_ = 0.9; ///< Safety factors

  int m_ = 30;
  int q_iop = 32;

  int k1 = 2;
  int mb, mx;
  PetscReal beta, avnorm;
  std::vector<Vec> Vm;
  arma::Mat<PetscReal> Hm;
  arma::Mat<PetscReal> F;
  Vec av = nullptr;

  Vec solution_tmp_ = nullptr;

  PetscReal t_now_tmp_ = 0.0;
  PetscReal t_step_ = 0.0;
  PetscReal t_step_next_ = 0.0;
  bool t_step_set_ = false;

  PetscReal tol_ = 1.0e-8;
  PetscReal btol_ = 1.0e-14;

  int krylov_stat_;

  int SetUpWorkSpace();

  int GenerateBasis(const Vec &v, int m);

  int AdvanceOneStep(const Vec &v);

  int GetDky(PetscReal t, int deg, Vec p_vec);

  // For logging events using PETSc LogEvent
  PetscLogEvent event_advance_one_step_;
  PetscLogEvent event_set_up_workspace_;
  PetscLogEvent event_generate_basis_;
  PetscLogEvent event_getdky_;
  PetscLogEvent event_free_workspace_;
};
}

#endif //PACMENSL_SRC_ODESOLVER_KRYLOVFSP_H_
