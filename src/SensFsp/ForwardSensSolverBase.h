//
// Created by Huy Vo on 2019-06-28.
//

#ifndef PACMENSL_SRC_SENSFSP_FORWARDSENSSOLVER_H_
#define PACMENSL_SRC_SENSFSP_FORWARDSENSSOLVER_H_

#include <sundials/sundials_nvector.h>
#include "Sys.h"

namespace pacmensl {
enum class ForwardSensType {
  CVODE
};

class ForwardSensSolverBase{
  using RhsFun = std::function<PacmenslErrorCode (PetscReal, Vec, Vec)>;
  using SensRhs1Fun = std::function<PacmenslErrorCode (int, PetscReal, Vec, Vec)>;
 public:

  explicit ForwardSensSolverBase(MPI_Comm new_comm);

  PacmenslErrorCode SetFinalTime(PetscReal _t_final);

  PacmenslErrorCode SetInitialSolution(Vec *_sol);

  PacmenslErrorCode SetRhs(std::function<PacmenslErrorCode (PetscReal, Vec, Vec)> _rhs);



  PacmenslErrorCode SetCurrentTime(PetscReal t);

  PacmenslErrorCode SetStatusOutput(int iprint);

  PacmenslErrorCode EnableLogging();

  PacmenslErrorCode SetStopCondition(const std::function<int(PetscReal, Vec, void *)> &stop_check_, void *stop_data_);

  PacmenslErrorCode EvaluateRHS(PetscReal t, Vec x, Vec y);

  virtual PacmenslErrorCode SetUp() {return 0;};

  virtual PetscInt Solve(); // Advance the solution_ toward final time. Return 0 if reaching final time, 1 if the Fsp criteria fails before reaching final time, -1 if fatal errors.

  PetscReal GetCurrentTime() const;

  virtual PacmenslErrorCode FreeWorkspace() { solution_ = nullptr; return 0;};

  ~OdeSolverBase();
};
}
#endif //PACMENSL_SRC_SENSFSP_FORWARDSENSSOLVER_H_
