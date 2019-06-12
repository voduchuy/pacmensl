//
// Created by Huy Vo on 2019-06-12.
//

#ifndef PECMEAL_SRC_ODESOLVER_KRYLOVFSP_H_
#define PECMEAL_SRC_ODESOLVER_KRYLOVFSP_H_

#include "OdeSolverBase.h"

namespace pecmeal{
class KrylovFsp : public OdeSolverBase {
 protected:

  PetscReal delta_ = 1.2, gamma_ = 0.9; ///< Safety factors

  int m;
  int q_iop = 2;
  struct KrylovBasisData{
    std::vector<Vec> Vm;
    arma::Mat<PetscReal> Hm;
  } basis_data_;

  Vec solution_tmp_;

  PetscReal t_now_tmp_;
  PetscReal t_step_;

  PetscReal tol_ = 1.0e-8;

  int krylov_stat_;

  int GenerateBasis(const Vec& v, int m);

  int AdvanceOneStep(const Vec& v);

  int GetDky(PetscReal t, int deg, Vec p_vec);
 public:

  explicit KrylovFsp(MPI_Comm comm);

  PetscInt solve( ) override;

  void free() override;
  ~KrylovFsp();
};
}

#endif //PECMEAL_SRC_ODESOLVER_KRYLOVFSP_H_
