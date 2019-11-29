//
// Created by Huy Vo on 5/29/18.
//

#ifndef PACMENSL_FSP_H
#define PACMENSL_FSP_H

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
#include"TsFsp.h"
#include"Sys.h"
#include"PetscWrap.h"

namespace pacmensl {
struct FspSolverComponentTiming
{
  PetscReal StatePartitioningTime;
  PetscReal MatrixGenerationTime;
  PetscReal ODESolveTime;
  PetscReal SolutionScatterTime;
  PetscReal RHSEvalTime;
  PetscReal TotalTime;
};

class FspSolverMultiSinks
{
  using Real = PetscReal;
  using Int = PetscInt;
 public:

  NOT_COPYABLE_NOT_MOVABLE(FspSolverMultiSinks);

  explicit FspSolverMultiSinks(MPI_Comm _comm, PartitioningType _part_type = PartitioningType::GRAPH,
                               ODESolverType _solve_type = CVODE);

  PacmenslErrorCode SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr, void *args);

  PacmenslErrorCode SetInitialBounds(arma::Row<int> &_fsp_size);

  PacmenslErrorCode SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

  PacmenslErrorCode SetModel(Model &model);

  PacmenslErrorCode SetInitialDistribution(const arma::Mat<Int> &_init_states, const arma::Col<PetscReal> &_init_probs);

  PacmenslErrorCode SetUp();

  PacmenslErrorCode SetFromOptions();

  PacmenslErrorCode SetLogging(PetscBool logging);

  PacmenslErrorCode SetVerbosity(int verbosity_level);

  PacmenslErrorCode SetLoadBalancingMethod(PartitioningType part_type);

  PacmenslErrorCode SetOdesType(ODESolverType odes_type);

  std::shared_ptr<const StateSetBase> GetStateSet();

  std::shared_ptr<OdeSolverBase> GetOdeSolver();

  FspSolverComponentTiming GetAvgComponentTiming();

  FiniteProblemSolverPerfInfo GetSolverPerfInfo();

  DiscreteDistribution Solve(PetscReal t_final, PetscReal fsp_tol = -1.0, PetscReal t_init = 0.0);

  std::vector<DiscreteDistribution> SolveTspan(const std::vector<PetscReal> &tspan,
                                               PetscReal fsp_tol = -1.0,
                                               PetscReal t_init = 0.0);

  PacmenslErrorCode ClearState();

  ~FspSolverMultiSinks();

 protected:

  MPI_Comm comm_ = nullptr;
  int      my_rank_;
  int      comm_size_;

  PartitioningType     partitioning_type_ = PartitioningType::BLOCK;
  PartitioningApproach repart_approach_   = PartitioningApproach::REPARTITION;
  ODESolverType        odes_type_         = CVODE;

  std::shared_ptr<StateSetConstrained>  state_set_ = nullptr;
  std::shared_ptr<FspMatrixConstrained> A_ = nullptr;
  std::shared_ptr<OdeSolverBase>        ode_solver_ = nullptr;
  std::shared_ptr<Petsc<Vec>>           p_ = nullptr;

  bool set_up_ = false;

  Model                                   model_;

  std::function<int(PetscReal, Vec, Vec)> tmatvec_;

  arma::Mat<Int>       init_states_;
  arma::Col<PetscReal> init_probs_;

  int verbosity_ = 0;

  bool                have_custom_constraints_ = false;
  fsp_constr_multi_fn fsp_constr_funs_;
  void *fsp_constr_args_ = nullptr;
  arma::Row<int>  fsp_bounds_;
  arma::Row<Real> fsp_expasion_factors_;

  // For error checking and expansion parameters
  PacmenslErrorCode CheckFspTolerance_(PetscReal t, Vec p, PetscReal &tol_exceed);

  virtual void set_expansion_parameters_() {};
  PetscReal fsp_tol_ = 1.0;
  PetscReal t_final_ = 0.0;
  PetscReal t_now_   = 0.0;

  PetscReal ode_rtol_ = 1.0e-6;
  PetscReal ode_atol_ = 1.0e-14;

  arma::Row<PetscReal> sinks_;
  arma::Row<int>       to_expand_;

  DiscreteDistribution Advance_(PetscReal t_final, PetscReal fsp_tol);
  PacmenslErrorCode MakeDiscreteDistribution_(DiscreteDistribution &dist);

  // For logging events using PETSc LogEvent
  PetscBool     logging_enabled = PETSC_FALSE;
  PetscLogEvent StateSetPartitioning;
  PetscLogEvent MatrixGeneration;
  PetscLogEvent ODESolve;
  PetscLogEvent SolutionScatter;
  PetscLogEvent RHSEvaluation;
  PetscLogEvent SettingUp;
  PetscLogEvent Solving;

 public:
  PacmenslErrorCode SetOdeTolerances(PetscReal rel_tol, PetscReal abs_tol);
};
}

#endif //PACMENSL_FSP_H
