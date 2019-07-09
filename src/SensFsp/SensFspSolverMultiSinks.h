//
// Created by Huy Vo on 2019-06-28.
//

#ifndef PACMENSL_SRC_SENSFSP_SENSFSPSOLVERMULTISINKS_H_
#define PACMENSL_SRC_SENSFSP_SENSFSPSOLVERMULTISINKS_H_

#include "SensModel.h"
#include "SensDiscreteDistribution.h"
#include "SensFspMatrix.h"
#include "ForwardSensCvodeFsp.h"

namespace pacmensl {
class SensFspSolverMultiSinks
{
 public:
  NOT_COPYABLE_NOT_MOVABLE(SensFspSolverMultiSinks);

  explicit SensFspSolverMultiSinks(MPI_Comm
                                   _comm,
                                   PartitioningType _part_type = PartitioningType::GRAPH,
                                   ODESolverType
                                   _solve_type = CVODE_BDF
  );

  PacmenslErrorCode SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr, void *args);
  PacmenslErrorCode SetInitialBounds(arma::Row<int> &_fsp_size);
  PacmenslErrorCode SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);
  PacmenslErrorCode SetModel(SensModel &model);
  PacmenslErrorCode SetVerbosity(int verbosity_level);

  PacmenslErrorCode SetInitialDistribution(const arma::Mat<pacmensl::Int> &_init_states,
                                           const arma::Col<PetscReal> &_init_probs,
                                           const std::vector<arma::Col<PetscReal>> &_init_sens);

  PacmenslErrorCode SetLoadBalancingMethod(PartitioningType part_type);
  PacmenslErrorCode SetOdesType(ForwardSensType odes_type);
  PacmenslErrorCode SetUp();
  const StateSetBase *GetStateSet();

  SensDiscreteDistribution Solve(PetscReal t_final, PetscReal fsp_tol);
  std::vector<SensDiscreteDistribution> SolveTspan(const std::vector<PetscReal> &tspan, PetscReal fsp_tol);

  PacmenslErrorCode ClearState();

  ~SensFspSolverMultiSinks();
 protected:

  MPI_Comm comm_ = nullptr;
  int      my_rank_;
  int      comm_size_;

  PartitioningType     partitioning_type_ = PartitioningType::GRAPH;
  PartitioningApproach repart_approach_   = PartitioningApproach::REPARTITION;
  ForwardSensType      sens_solver_type   = ForwardSensType::CVODE;

  Petsc<Vec>              p_;
  std::vector<Petsc<Vec>> dp_;

  std::shared_ptr<StateSetConstrained>                 state_set_;
  std::shared_ptr<SensFspMatrix<FspMatrixConstrained>> A_;
  std::shared_ptr<ForwardSensCvodeFsp>                 sens_solver_;

  bool                                                 set_up_ = false;

  SensModel                                    model_;

  std::function<int(PetscReal, Vec, Vec)>      matvec_;
  std::function<int(int, PetscReal, Vec, Vec)> dmatvec_;

  arma::Mat<Int>                    init_states_;
  arma::Col<PetscReal>              init_probs_;
  std::vector<arma::Col<PetscReal>> init_sens_;

  int verbosity_ = 0;

  bool                have_custom_constraints_ = false;
  fsp_constr_multi_fn fsp_constr_funs_;
  void* fsp_constr_args_ = nullptr;
  arma::Row<int>      fsp_bounds_;

  arma::Row<Real> fsp_expasion_factors_;
  virtual void set_expansion_parameters_() {};


  Real fsp_tol_ = 1.0;
  Real t_final_ = 0.0;
  Real t_now_   = 0.0;
  arma::Row<PetscReal> sinks_;

  arma::Row<int>       to_expand_;
  SensDiscreteDistribution Advance_(PetscReal t_final, PetscReal fsp_tol);
  int CheckFspTolerance_(PetscReal t, Vec p);

  PacmenslErrorCode MakeSensDiscreteDistribution_(SensDiscreteDistribution &dist);
};
}
#endif //PACMENSL_SRC_SENSFSP_SENSFSPSOLVERMULTISINKS_H_
