//
// Created by Huy Vo on 12/6/18.
//

#ifndef PACMENSL_ODESOLVERBASE_H
#define PACMENSL_ODESOLVERBASE_H

#include <sundials/sundials_nvector.h>
#include "cme_util.h"

namespace pacmensl {
enum ODESolverType { KRYLOV, CVODE_BDF };

struct FiniteProblemSolverPerfInfo {
  PetscInt n_step;
  std::vector<PetscInt> n_eqs;
  std::vector<PetscLogDouble> cpu_time;
  std::vector<PetscReal> model_time;
};

class OdeSolverBase {
 protected:
  MPI_Comm comm_ = MPI_COMM_NULL;
  int my_rank_;
  int comm_size_;

  Vec *solution_ = nullptr;
  std::function<void(PetscReal t, Vec x, Vec y)> rhs_;
  PetscReal t_now_ = 0.0;
  PetscReal t_final_ = 0.0;

  // For logging and monitoring
  int print_intermediate = 0;

  /*
   * Function to check early stopping condition.
   */
  std::function<int(PetscReal t, Vec p, void *data)> stop_check_ = nullptr;
  void *stop_data_ = nullptr;

  PetscBool logging_enabled = PETSC_FALSE;

  FiniteProblemSolverPerfInfo perf_info;

 public:

  explicit OdeSolverBase(MPI_Comm new_comm);

  void SetFinalTime(PetscReal _t_final);

  void set_initial_solution(Vec *sol0);

  void set_rhs(std::function<void(PetscReal, Vec, Vec)> _rhs);

  void set_current_time(PetscReal t);

  void set_print_intermediate(int iprint);

  void EnableLogging();

  void SetStopCondition(const std::function<int(PetscReal, Vec, void *)> &stop_check_, void *stop_data_);

  void EvaluateRHS(PetscReal t, Vec x, Vec y);

  virtual void SetUp() {};

  virtual PetscInt Solve(); // Advance the solution_ toward final time. Return 0 if reaching final time, 1 if the FSP criteria fails before reaching final time.

  PetscReal GetCurrentTime() const;

  FiniteProblemSolverPerfInfo GetAvgPerfInfo() const;

  virtual void FreeWorkspace() { solution_ = nullptr; };

  ~OdeSolverBase();
};
}

#endif //PACMENSL_ODESOLVERBASE_H
