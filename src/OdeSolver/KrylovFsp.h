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

  int SetUp() override;

  PetscInt Solve() override;

  PacmenslErrorCode SetOrthLength(int q);

  int FreeWorkspace() override;

  ~KrylovFsp();

 protected:

  const int max_reject_ = 10000;
  PetscReal delta_ = 1.2, gamma_ = 0.9; ///< Safety factors

  int m_min_ = 15, m_max_ = 60, m_next_ = 30;
  int m_ = 30;
  int q_iop = -1;

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

  PetscReal btol_ = 1.0e-14;

  int krylov_stat_ = 0;

  int SetUpWorkSpace();

  int GenerateBasis(const Vec &v,int m_start, PetscBool *happy_breakdown);

  int AdvanceOneStep(const Vec &v);

  int GetDky(PetscReal t, int deg, Vec p_vec);

  inline int EstimateCost_(PetscReal tau_new,PetscInt m_new, PetscInt *cost);

  // For logging events using PETSc LogEvent
  PetscLogEvent event_advance_one_step_;
  PetscLogEvent event_set_up_workspace_;
  PetscLogEvent event_generate_basis_;
  PetscLogEvent event_getdky_;
  PetscLogEvent event_free_workspace_;
};
}

#endif //PACMENSL_SRC_ODESOLVER_KRYLOVFSP_H_
