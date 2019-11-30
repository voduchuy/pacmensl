//
// Created by Huy Vo on 12/6/18.
//

#ifndef PACMENSL_EPICFSP_H
#define PACMENSL_EPICFSP_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include<nvector/nvector_petsc.h>
#include "EpicHeaders/Epic.h"
#include "OdeSolverBase.h"
#include "StateSetConstrained.h"
#include "Sys.h"



namespace pacmensl {

// Implement the "one-step" mode for the exponential integrator
// This is a child class of the EpiRK5V in EPIC.
class EpiRK4SVInterface: public EpiRK4SV {
 public:

  // We only need one kind of constructor for the CME
  explicit EpiRK4SVInterface(CVRhsFn f, CVSpilsJacTimesVecFn jtv, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);

  PacmenslErrorCode Step(const realtype hStart,
                         const realtype hMax,
                         const realtype absTol,
                         const realtype relTol,
                         const realtype t0,
                         const realtype tFinal,
                         const int numBands,
                         int *startingBasisSizes,
                         N_Vector y,
                         realtype *tnew,
                         realtype *hnew);
};

// The following class takes care of interfacing the adaptive FSP solver with the exponential integrators implemented in EPIC.
class EpicFsp : public OdeSolverBase {
 public:
  explicit EpicFsp(MPI_Comm _comm,float num_bands);

  PacmenslErrorCode SetUp() override ;

  PetscInt Solve() override;

  int FreeWorkspace() override;

  ~EpicFsp() override;
 protected:
  EpiRK4SVInterface* epic_stepper = nullptr;
  N_Vector solution_wrapper = nullptr;
  N_Vector solution_tmp = nullptr;

  const int min_krylov_size_ = 25;
  const int max_krylov_size_ = 50;

  float num_bands_;

  PetscReal t_now_tmp = 0.0;
  static int epic_rhs(double t,N_Vector u,N_Vector udot,void *solver);
  static int epic_jac(N_Vector v,N_Vector Jv,realtype t,
                      N_Vector u,N_Vector fu,
                      void *FPS_ptr,N_Vector tmp);

};
}

#endif //PACMENSL_EPICFSP_H
