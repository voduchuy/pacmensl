//
// Created by Huy Vo on 5/29/18.
//

#ifndef PECMEAL_FSP_H
#define PECMEAL_FSP_H

#include<algorithm>
#include<cstdlib>
#include<cmath>
#include"Model.h"
#include"DiscreteDistribution.h"
#include"FspMatrixBase.h"
#include"FspMatrixConstrained.h"
#include"OdeSolverBase.h"
#include"StateSetBase.h"
#include"StateSetConstrained.h"
#include"KrylovFsp.h"
#include"CvodeFsp.h"
#include"cme_util.h"

namespace pecmeal {
struct FspSolverComponentTiming {
  PetscReal StatePartitioningTime;
  PetscReal MatrixGenerationTime;
  PetscReal ODESolveTime;
  PetscReal SolutionScatterTime;
  PetscReal RHSEvalTime;
  PetscReal TotalTime;
};

class FspSolverBase {
  using Real = PetscReal;
  using Int = PetscInt;
 public:
  NOT_COPYABLE_NOT_MOVABLE(FspSolverBase);

  explicit FspSolverBase(MPI_Comm _comm, PartitioningType _part_type = Graph, ODESolverType _solve_type = CVODE_BDF);

  void SetConstraintFunctions(fsp_constr_multi_fn *lhs_constr);

  void SetInitialBounds(arma::Row<int> &_fsp_size);

  void SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

  void SetModel(Model &model);

  void SetVerbosity(int verbosity_level);

  void SetInitialDistribution(const arma::Mat<Int> &_init_states, const arma::Col<PetscReal> &_init_probs);

  void SetLogging(PetscBool logging);

  void SetFromOptions();

  void SetPartitioningMethod(PartitioningType part_type);

  void SetOdesType(ODESolverType odes_type);

  void SetUp();

  Vec &GetP();

  const StateSetBase *GetStateSet();

  FspSolverComponentTiming GetAvgComponentTiming();

  FiniteProblemSolverPerfInfo GetSolverPerfInfo();

  DiscreteDistribution Solve(PetscReal t_final, PetscReal fsp_tol);

  std::vector<DiscreteDistribution> SolveTspan(const arma::Row<PetscReal> &tspan, PetscReal fsp_tol);

  void Destroy();

  ~FspSolverBase();
 private:

  MPI_Comm comm_ = nullptr;
  int my_rank_;
  int comm_size_;

  PartitioningType partitioning_type_ = Graph;
  PartitioningApproach repart_approach_ = Repartition;
  ODESolverType odes_type_ = CVODE_BDF;

  StateSetBase *state_set_ = nullptr;
  Vec *p_ = nullptr;
  FspMatrixBase *A_ = nullptr;
  OdeSolverBase *ode_solver_ = nullptr;

  Model model_;

  std::function<void(PetscReal, Vec, Vec)> tmatvec_;

  arma::Mat<Int> init_states_;
  arma::Col<PetscReal> init_probs_;

  int verbosity_ = 0;
  bool have_custom_constraints_ = false;

  fsp_constr_multi_fn *fsp_constr_funs_;
  arma::Row<int> fsp_bounds_;
  arma::Row<Real> fsp_expasion_factors_;

  // For error checking and expansion parameters
  int CheckFspTolerance_(PetscReal t, Vec p);

  virtual void set_expansion_parameters_() {};
  Real fsp_tol_ = 1.0;
  Real t_final_ = 0.0;

  arma::Row<PetscReal> sinks_;
  arma::Row<int> to_expand_;

  DiscreteDistribution Advance_(PetscReal t_final, PetscReal fsp_tol);

  // For logging events using PETSc LogEvent
  PetscBool logging_enabled = PETSC_FALSE;
  PetscLogEvent StateSetPartitioning;
  PetscLogEvent MatrixGeneration;
  PetscLogEvent ODESolve;
  PetscLogEvent SolutionScatter;
  PetscLogEvent RHSEvaluation;
  PetscLogEvent SettingUp;
  PetscLogEvent Solving;
};
}

#endif //PECMEAL_FSP_H
