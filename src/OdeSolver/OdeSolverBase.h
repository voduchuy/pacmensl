//
// Created by Huy Vo on 12/6/18.
//

#ifndef PACMENSL_ODESOLVERBASE_H
#define PACMENSL_ODESOLVERBASE_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include<nvector/nvector_petsc.h>
#include "Sys.h"
#include "StateSetConstrained.h"
#include "FspMatrixBase.h"

namespace pacmensl {
enum ODESolverType { KRYLOV, CVODE, PETSC, EPIC };

struct FiniteProblemSolverPerfInfo {
  PetscInt n_step;
  std::vector<PetscInt> n_eqs;
  std::vector<PetscLogDouble> cpu_time;
  std::vector<PetscReal> model_time;
};

class OdeSolverBase {
 public:

  explicit OdeSolverBase(MPI_Comm new_comm);

  PacmenslErrorCode SetFinalTime(PetscReal _t_final);

  PacmenslErrorCode SetInitialSolution(Vec *_sol);

  PacmenslErrorCode SetFspMatPtr(FspMatrixBase* mat);

  PacmenslErrorCode SetRhs(std::function<PacmenslErrorCode (PetscReal, Vec, Vec)> _rhs);

  int SetTolerances(PetscReal _r_tol, PetscReal _abs_tol);

  PacmenslErrorCode SetCurrentTime(PetscReal t);

  PacmenslErrorCode SetStatusOutput(int iprint);

  PacmenslErrorCode EnableLogging();

  PacmenslErrorCode SetStopCondition(const std::function<PacmenslErrorCode (PetscReal, Vec, PetscReal&, void *)> &stop_check_, void *stop_data_);

  PacmenslErrorCode EvaluateRHS(PetscReal t, Vec x, Vec y);

  virtual PacmenslErrorCode SetUp() {return 0;};

  virtual PetscInt Solve(); // Advance the solution_ toward final time. Return 0 if reaching final time, 1 if the Fsp criteria fails before reaching final time, -1 if fatal errors.

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
};
}

#endif //PACMENSL_ODESOLVERBASE_H
