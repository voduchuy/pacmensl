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

#ifndef PACMENSL_SRC_SENSFSP_FORWARDSENSSOLVER_H_
#define PACMENSL_SRC_SENSFSP_FORWARDSENSSOLVER_H_

#include <sundials/sundials_nvector.h>
#include "OdeSolverBase.h"
#include "Sys.h"
#include "PetscWrap.h"

namespace pacmensl {
enum class ForwardSensType {
  CVODE,
  KRYLOV,
};

class ForwardSensSolverBase {
  using RhsFun = std::function<PacmenslErrorCode(PetscReal, Vec, Vec)>;
  using SensRhs1Fun = std::function<PacmenslErrorCode(int, PetscReal, Vec, Vec)>;
 public:

  NOT_COPYABLE_NOT_MOVABLE(ForwardSensSolverBase);

  explicit ForwardSensSolverBase(MPI_Comm new_comm);

  PacmenslErrorCode SetFinalTime(PetscReal _t_final);

  PacmenslErrorCode SetInitialSolution(Petsc<Vec> &sol);

  PacmenslErrorCode SetInitialSensitivity(std::vector<Petsc < Vec>> &sens_vecs);

  PacmenslErrorCode SetRhs(RhsFun rhs);

  PacmenslErrorCode SetSensRhs(SensRhs1Fun sensrhs);

  PacmenslErrorCode SetCurrentTime(PetscReal t);

  PacmenslErrorCode SetStatusOutput(int iprint);

  PacmenslErrorCode EnableLogging();

  PacmenslErrorCode SetStopCondition(const std::function<int(PetscReal, Vec, int, Vec *, void *)> &stop_check,
                                     void *stop_data);

  virtual PacmenslErrorCode SetUp() { return 0; }

  virtual PetscInt Solve() { return 0; }; // Advance the solution_ toward final time. Return 0 if reaching final time, 1 if the Fsp criteria fails before reaching final time, -1 if fatal errors.

  virtual PacmenslErrorCode FreeWorkspace();

  PetscReal GetCurrentTime() const { return t_now_; };

  virtual ~ForwardSensSolverBase();

  int EvaluateRHS(PetscReal t, Vec x, Vec y) { return rhs_(t, x, y); };

  int EvaluateSensRHS(int iS, PetscReal t, Vec x, Vec y) { return srhs_(iS, t, x, y); };
 protected:
  MPI_Comm comm_ = MPI_COMM_NULL;
  int      my_rank_;
  int      comm_size_;

  bool set_up_ = false;

  Vec                *solution_      = nullptr;
  std::vector<Vec *> sens_vecs_;
  int                num_parameters_ = 0;

  RhsFun      rhs_;
  SensRhs1Fun srhs_;

  PetscReal t_now_   = 0.0;
  PetscReal t_final_ = 0.0;

  // For logging and monitoring
  int                                                    print_intermediate = 0;

  /*
   * Function to check early stopping condition.
   */
  std::function<int(PetscReal, Vec, int, Vec *, void *)> stop_check_        = nullptr;
  void                                                   *stop_data_        = nullptr;

  PetscBool logging_enabled = PETSC_FALSE;
  FiniteProblemSolverPerfInfo perf_info;
};
}
#endif //PACMENSL_SRC_SENSFSP_FORWARDSENSSOLVER_H_
