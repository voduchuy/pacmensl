/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/**
 * @file FspSolverMultiSinks.h
 */
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

/**
 * @brief Performance logging for FSP solver.
 */
struct FspSolverComponentTiming
{
  PetscReal StatePartitioningTime; ///< State space partitioning
  PetscReal MatrixGenerationTime; ///< Transition-rate matrix generation
  PetscReal ODESolveTime; ///< Time spent in ODE solver
  PetscReal SolutionScatterTime; ///< VecScatter calls
  PetscReal RHSEvalTime; ///< Matrix-vector multiplication
  PetscReal TotalTime; ///< Total time
  PetscReal TotalFlops; ///< Total number of FLOPs
};

/**
 * @brief Finite State Projection (FSP) solver using an adaptively truncated state space with multiple sink states defined through inequality constraints.
 */
class FspSolverMultiSinks
{
  using Real = PetscReal;
  using Int = PetscInt;
  
 public:

  NOT_COPYABLE_NOT_MOVABLE(FspSolverMultiSinks);

  /**
   * @brief Constructor for the FSP solver.
   * @param _comm (in) Communicator.
   * @param _part_type (in) Load-balancing method for the state space.
   * @param _solve_type (in) ODE solver type. Default is CVODE.
   */
  explicit FspSolverMultiSinks(MPI_Comm _comm, PartitioningType _part_type = PartitioningType::GRAPH,
                               ODESolverType _solve_type = CVODE);

  /**
   * @brief Set the inequality constraints for the FSP states.
   * @param lhs_constr (in) Function to evaluate the left-hand-side functions in the inequalities.
   * @param args (in) pointer to additional data structure if needed.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr, void *args);

  /**
   * @brief Set the intial bounds (the numbers on the right hand sides of the inequality constraints) for the FSP.
   * @param _bounds (in) vector of bounds.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetInitialBounds(arma::Row<int> &_bounds);

  /**
   * @brief Set the multiplicative factors for the expansion.
   * @details When expansion is needed, the existing constraint bounds will be multiply by the factors of `(1 + factors)`.
   * @param _expansion_factors (in) vector of factors.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

  /**
   * @brief Set the stochastic reaction network model.
   * @param model (in) the model.
   * @return Error: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetModel(Model &model);

  /**
   * @brief Set the initial solution of the CME.
   * @details This method is __collective__.
   * @attention We do not check whether the probabilities are positive nor sum to one. This responsibility rests solely on the user.
   * @param _init_states (in) matrix of initial states, arranged column-wise.
   * @param _init_probs (in) vector of initial probabilities.
   * @return Error: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetInitialDistribution(const arma::Mat<Int> &_init_states, const arma::Col<PetscReal> &_init_probs);

  /**
   * @brief Set the initial solution of the CME using an existing \ref DiscreteDistribution object.
   * @param init_dist (in) the initial distribution.
   * @return Error: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetInitialDistribution(DiscreteDistribution& init_dist);

  /**
   * @brief Initialize all data structures needed for calling \ref Solve or \ref SolveTSpan.
   * @return Error: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetUp();

  /**
   * @brief Set solver attributes based on command-line input arguments.
   * @return Error: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetFromOptions();

  /**
   * @brief Enable/disable performance logging.
   * @details This method is __collective__, meaning that it must be called by all owning processes.
   * @param logging (in) whether to enable logging (true/false).
   * @return Error: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetLogging(PetscBool logging);

  /**
   * @brief Set verbosity level.
   * @param verbosity_level (in) degree of verbosity (0, 1, 2):\n
   * 0: no printing to the screen.
   * 1: print every time the FSP state space is expanded.
   * 2: print after every successful ODE time step.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetVerbosity(int verbosity_level);

  /**
   * @brief Set load-balancing method for distributing the state space.
   * @param part_type (in) partitioning method.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetLoadBalancingMethod(PartitioningType part_type);

  /**
   * @brief Set the ODE solver.
   * @details This method is __collective__. All owning processes must give the same input.
   * @param odes_type (in) ODE solver type.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetOdesType(ODESolverType odes_type);

  /**
   * @brief Select the ODE method used by PETSc TS integrator.
   * @details This method is __collective__. All owning processes must give the same input. \n
   * @details This method has no effect if the underlying ODE solver does not use PETSc.
   * @param ts_type (in) name of PETSC TS solver. See PETSc's user manual for the list of methods.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetOdesPetscType(std::string ts_type);

  /**
   * @brief Choose the orthogonalization length if the ODE solver chosen is Krylov wit incomplete orthogonalization.
   * @details This method is __collective__. All owning processes must give the same input. \n
   * @details This method has no effect if the underlying ODE solver does not use Krylov.
   * @param q (in) orthogonalization length.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode SetKrylovOrthLength(int q);
  
  /**
   * @brief Choose the range of Krylov basis sizes the ODE solver chosen is Krylov wit incomplete orthogonalization.
   * @details This method is __collective__. All owning processes must give the same input. \n
   * @details This method has no effect if the underlying ODE solver does not use Krylov.
   * @param m_min (in) minimum Krylov basis size.
   * @param m_max (in) maximum Krylov basis size.
   * @return Error code: 0 (success) or -1 (failure).
  */
  PacmenslErrorCode SetKrylovDimRange(int m_min, int m_max);

  /**
   * @brief Get a pointer to the states of the final FSP state space.
   * @return Pointer to the internal \ref StateSetBase object.
   */
  std::shared_ptr<const StateSetBase> GetStateSet();

  /**
   * @brief Get pointer to the internal ODE solver.
   * @return Pointer to the internal \ref OdeSolverBase object.
   */
  std::shared_ptr<OdeSolverBase> GetOdeSolver();

  /**
   * @brief Reduce component timing across owning processes to obtain global performance statistics.
   * @details This method is __collective__, meaning that it must be called by all owning processes.
   * @param op (in) operation ("min", "max", "sum").
   * @return Component timing object.
   */
  FspSolverComponentTiming ReduceComponentTiming(char *op);

  /**
   * @brief Get local logging information.
   * @return Local logging information.
   */
  FiniteProblemSolverPerfInfo GetSolverPerfInfo();

  /**
   * @brief Solve the CME and output the solution at a specified time.
   * @param t_final (in) final time.
   * @param fsp_tol (in) FSP error tolerance.
   * @param t_init (in, optional) initial time. Default is 0.
   * @return
   */
  DiscreteDistribution Solve(PetscReal t_final, PetscReal fsp_tol = -1.0, PetscReal t_init = 0.0);

  /**
   * @brief Solve the CME and output solutions at specified times.
   * @param tspan (in) vector of output times.
   * @param fsp_tol (in) FSP error tolerance.
   * @param t_init (in, optional) initial time. Default is 0.
   * @return `std::vector` containing the solutions at specified times as \ref DiscreteDistribution objects.
   */
  std::vector<DiscreteDistribution> SolveTspan(const std::vector<PetscReal> &tspan,
                                               PetscReal fsp_tol = -1.0,
                                               PetscReal t_init = 0.0);
  
  /**
   * @brief Clear the internal state of the object and free up all memories allocated.
   * @details This method is __collective__, meaning that it must be called by all owning processes.
   * @return Error code: 0 (success) or -1 (failure).
   */
  PacmenslErrorCode ClearState();

  ~FspSolverMultiSinks();

 protected:

  MPI_Comm comm_ = MPI_COMM_NULL;
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

  bool                has_custom_constraints_ = false;
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

  // Cache options for ODE solvers
  bool custom_ts_type_ = false;
  std::string ts_type_ = "";

  bool custom_krylov_ = false;
  int q_iop_ = -1;
  int m_min_ = 25, m_max_ = 60;

 public:
  /**
   * @brief Set tolerance for ODE solver.
   * @details This method is __collective__. All owning processes must give the same input. \n
   * @param rel_tol (in) relative tolerance.
   * @param abs_tol (out) absolute tolerance.
   * @return Error code: 0 (success) or -1 (failure)
   */
  PacmenslErrorCode SetOdeTolerances(PetscReal rel_tol, PetscReal abs_tol);
};
}

#endif //PACMENSL_FSP_H
