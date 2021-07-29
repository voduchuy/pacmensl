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

#ifndef PACMENSL_CVODEFSP_H
#define PACMENSL_CVODEFSP_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include<nvector/nvector_petsc.h>
#include "OdeSolverBase.h"
#include "StateSetConstrained.h"
#include "Sys.h"


namespace pacmensl {
class CvodeFsp : public OdeSolverBase {
 public:
  explicit CvodeFsp(MPI_Comm _comm, int lmm = CV_BDF);

  PacmenslErrorCode SetUp() override ;

  PetscInt Solve() override;

  int FreeWorkspace() override;

  ~CvodeFsp();
 protected:
  int lmm_ = CV_BDF;
  void *cvode_mem = nullptr;
  SUNLinearSolver linear_solver = nullptr;
  N_Vector solution_wrapper = nullptr;
  N_Vector solution_tmp = nullptr;
  N_Vector constr_vec_ = nullptr;

  PetscReal t_now_tmp = 0.0;
  int cvode_stat = 0;
  static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *solver);
  static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                       N_Vector u, N_Vector fu,
                       void *FPS_ptr, N_Vector tmp);
};
}

#endif //PACMENSL_CVODEFSP_H
