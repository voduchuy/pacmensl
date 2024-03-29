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

#ifndef PACMENSL_ODESOLVERBASE_H
#define PACMENSL_ODESOLVERBASE_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include "Sys.h"
#include "StateSetConstrained.h"
#include "FspMatrixBase.h"
#include "FspMatrixConstrained.h"

namespace pacmensl {
enum ODESolverType { KRYLOV, CVODE, PETSC, EPIC };

struct FiniteProblemSolverPerfInfo {
  PetscInt n_step;
  std::vector<PetscInt> n_eqs;
  std::vector<PetscLogDouble> cpu_time;
  std::vector<PetscReal> model_time;
};

/**
 * @brief Base class for the ODE solver.
 * @details The ODE solver is used to integrate the finite-state system after the state space has been truncated by the FSP.
 */
class OdeSolverBase {
 public:

  /**
   * @brief Constructor.
   * @param new_comm (in) communicator context.
   */
  explicit OdeSolverBase(MPI_Comm new_comm);

  /**
   * @brief Set final time of the ODE integrator.
   * @details The method is __collective__, meaning that it must be called by all processes that own this object.
   * @attention All owning processes must give the same inputs.
   * @param _t_final (in) final time.
   * @return Error code: 0 (success), -1 (failure).
   */
  PacmenslErrorCode SetFinalTime(PetscReal _t_final);

  /**
   * @brief Set the intial solution.
   * @details The method is __collective__, meaning that it must be called by all processes that own this object.
   * @param _sol (in) the initial solution.
   * @return Error code: 0 (success), -1 (failure).
   */
  PacmenslErrorCode SetInitialSolution(Vec *_sol);

  /**
   * @brief Set pointer to the FSP matrix.
   * @param mat (in) pointer to the FSP matrix object.
   * @return Error code: 0 (success), -1 (failure).
   */
  PacmenslErrorCode SetFspMatPtr(FspMatrixBase* mat);

  /**
   * @brief Set function to evaluate ODE right-hand side.
   * @param _rhs (in) RHS function.
   * @return Error code: 0 (success), -1 (failure).
   */
  PacmenslErrorCode SetRhs(std::function<PacmenslErrorCode(PetscReal,Vec,Vec)> _rhs);

  /**
   * @brief Set error tolerances.
   * @details The method is __collective__, meaning that it must be called by all processes that own this object.
   * @attention All owning processes must give the same inputs.
   * @param _r_tol (in) relative tolerance.
   * @param _abs_tol (in) absolute tolerance.
   * @return Error code: 0 (success), -1 (failure).
   */
  int SetTolerances(PetscReal _r_tol, PetscReal _abs_tol);

  PacmenslErrorCode SetCurrentTime(PetscReal t);

  PacmenslErrorCode SetStatusOutput(int iprint);

  PacmenslErrorCode EnableLogging();

  PacmenslErrorCode SetStopCondition(const std::function<PacmenslErrorCode (PetscReal, Vec, PetscReal&, void *)> &stop_check_, void *stop_data_);

  PacmenslErrorCode EvaluateRHS(PetscReal t, Vec x, Vec y);

  virtual PacmenslErrorCode SetUp() {return 0;};

  virtual PetscInt Solve(); ///< Advance the solution_ toward final time. Return 0 if reaching final time, 1 if the Fsp criteria fails before reaching final time, -1 if fatal errors.

  PetscReal GetCurrentTime() const;

  FiniteProblemSolverPerfInfo GetAvgPerfInfo() const;

  virtual PacmenslErrorCode FreeWorkspace() { solution_ = nullptr; return 0;};
  virtual ~OdeSolverBase();
 protected:
  MPI_Comm comm_ = MPI_COMM_NULL;
  int my_rank_;
  int comm_size_;

  Vec *solution_ = nullptr;
  std::function<int (PetscReal t, Vec x, Vec y)> rhs_;
  int rhs_cost_loc_ = 0;

  FspMatrixBase* fspmat_;

  PetscReal t_now_ = 0.0;
  PetscReal t_final_ = 0.0;

  // For logging and monitoring
  int print_intermediate = 0;
  /*
   * Function to check early stopping condition.
   */
  std::function<PacmenslErrorCode (PetscReal t, Vec p, PetscReal& tol_exceed, void *data)> stop_check_ = nullptr;

  void *stop_data_ = nullptr;

  PetscBool logging_enabled           = PETSC_FALSE;

  FiniteProblemSolverPerfInfo perf_info;
  PetscReal                   rel_tol_ = 1.0e-6;
  PetscReal                   abs_tol_ = 1.0e-14;

  friend PacmenslErrorCode FspMatrixBase::GetLocalMVFlops(PetscInt * nflops);
  friend PacmenslErrorCode FspMatrixConstrained::GetLocalMVFlops(PetscInt *nflops);
};
}

#endif //PACMENSL_ODESOLVERBASE_H
