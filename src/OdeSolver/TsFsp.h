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

#ifndef PACMENSL_TSFSP_H
#define PACMENSL_TSFSP_H

#include "OdeSolverBase.h"
#include "PetscWrap.h"
#include "Sys.h"


namespace pacmensl {
class TsFsp: public OdeSolverBase {
 public:
  explicit TsFsp(MPI_Comm _comm);

  PacmenslErrorCode SetUp() override ;

  PetscInt Solve() override;

  PacmenslErrorCode SetTsType(std::string type);

  int FreeWorkspace() override;

  ~TsFsp() override;

 protected:
  std::string type_ = std::string(TSROSW);
  Petsc<TS> ts_;
  PetscReal t_now_tmp = 0.0;
  PetscInt fsp_stop_ = 0;
  Vec solution_tmp_;

  int njac = 0, nstep = 0;

  Mat J = nullptr;
  static int TSRhsFunc(TS ts, PetscReal t, Vec u, Vec F, void* ctx);
  static int TSJacFunc(TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx);
  static int TSIFunc(TS ts, PetscReal t, Vec u, Vec u_t, Vec F, void*ctx);
  static int TSIJacFunc(TS ts, PetscReal t, Vec u, Vec u_t, PetscReal a, Mat A, Mat P, void*ctx);
  static int TSCheckFspError(TS ts);

  static int CheckImplicitType(TSType type, int* implicit);
};
}

#endif //PACMENSL_TSFSP_H
