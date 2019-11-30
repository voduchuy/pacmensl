//
// Created by Huy Vo on 12/6/18.
//
#include "EpicFsp.h"
#include "OdeSolverBase.h"

namespace pacmensl {



EpicFsp::EpicFsp(MPI_Comm _comm,float num_bands) : OdeSolverBase(_comm),
num_bands_(num_bands)
{
}

PetscInt EpicFsp::Solve()
{
  PacmenslErrorCode ierr;
  PetscErrorCode    petsc_err;
  Vec               solution_tmp_dat = N_VGetVector_Petsc(solution_tmp);
  petsc_err = VecCopy(*solution_, solution_tmp_dat);
  CHKERRQ(petsc_err);
  realtype hstart = t_final_/100.0;
  realtype hmax = t_final_/100.0;
  realtype tnew;

  int start_krylov_sizes[] = {min_krylov_size_, min_krylov_size_, min_krylov_size_};
  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  int       stop         = 0;
  PetscReal error_excess = 0.0;
  while (t_now_ < t_final_)
  {
    ierr = epic_stepper->Step(hstart,
                              hmax,
                              abs_tol_,
                              rel_tol_,
                              t_now_,
                              t_final_,
                              num_bands_,
                              &start_krylov_sizes[0],
                              solution_tmp,
                              &tnew,
                              &hstart);
    CHKERRQ(ierr);

    t_now_tmp = (PetscReal) tnew;
    // Check that the temporary solution_ satisfies Fsp tolerance
    if (stop_check_ != nullptr)
    {
      ierr = stop_check_(t_now_tmp,solution_tmp_dat,error_excess,stop_data_);
      PACMENSLCHKERRQ(ierr);
      if (error_excess > 0.0)
      {
        return 1;
      }
    }

    t_now_ = t_now_tmp;
    if (print_intermediate)
    {
      PetscPrintf(comm_,"t_now_ = %.2e \n",t_now_);
    }
    if (logging_enabled)
    {
      perf_info.model_time[perf_info.n_step] = t_now_;
      petsc_err = VecGetSize(*solution_,&perf_info.n_eqs[size_t(perf_info.n_step)]);
      CHKERRQ(petsc_err);
      petsc_err = PetscTime(&perf_info.cpu_time[perf_info.n_step]);
      CHKERRQ(petsc_err);
      perf_info.n_step += 1;
    }
    // Copy data from temporary vector to solution_ vector
    petsc_err = VecCopy(solution_tmp_dat,*solution_);
  }
  CHKERRQ(petsc_err);
  return stop;
}

int EpicFsp::epic_rhs(double t,N_Vector u,N_Vector udot,void *solver)
{
  int ierr{0};
  Vec udata    = N_VGetVector_Petsc(u);
  Vec udotdata = N_VGetVector_Petsc(udot);
  ierr = (( pacmensl::OdeSolverBase * ) solver)->EvaluateRHS(t,udata,udotdata);
  PACMENSLCHKERRQ(ierr);
  return ierr;
}

int
EpicFsp::epic_jac(N_Vector v,N_Vector Jv,realtype t,N_Vector u,N_Vector fu,void *FPS_ptr,
                  N_Vector tmp)
{
  int ierr{0};
  Vec vdata  = N_VGetVector_Petsc(v);
  Vec Jvdata = N_VGetVector_Petsc(Jv);
  ierr = (( pacmensl::OdeSolverBase * ) FPS_ptr)->EvaluateRHS(t,vdata,Jvdata);
  PACMENSLCHKERRQ(ierr);
  return ierr;
}

int EpicFsp::FreeWorkspace()
{
  OdeSolverBase::FreeWorkspace();
  delete epic_stepper;
  if (solution_tmp != nullptr) N_VDestroy(solution_tmp);
  if (solution_wrapper != nullptr) N_VDestroy(solution_wrapper);
  solution_tmp     = nullptr;
  solution_wrapper = nullptr;
  epic_stepper = nullptr;
  return 0;
}

EpicFsp::~EpicFsp()
{
  FreeWorkspace();
}

PacmenslErrorCode EpicFsp::SetUp()
{
  // Make sure the necessary data has been set
  if (solution_ == nullptr) return -1;
  if (rhs_ == nullptr) return -1;

  PetscInt petsc_err;
  // N_Vector wrapper for the solution_
  solution_wrapper = N_VMake_Petsc(*solution_);

  // Copy solution_ to the temporary solution_
  solution_tmp = N_VClone(solution_wrapper);
  Vec solution_tmp_dat = N_VGetVector_Petsc(solution_tmp);
  petsc_err = VecCopy(*solution_,solution_tmp_dat);
  CHKERRQ(petsc_err);

  // Set CVODE starting time to the current timepoint
  t_now_tmp = t_now_;

  // Create Epic integrator
  PetscInt neq;
  VecGetSize(*solution_, &neq);
  epic_stepper = new EpiRK4SVInterface(&epic_rhs, &epic_jac, this, max_krylov_size_, solution_tmp, neq);

  if (epic_stepper == nullptr){
    PetscPrintf(comm_, "Failure in constructing EPIC integrator.\n");
    return -1;
  }

  return 0;
}



EpiRK4SVInterface::EpiRK4SVInterface(CVRhsFn f,
                               CVSpilsJacTimesVecFn jtv,
                               void *userData,
                               int maxKrylovIters,
                               N_Vector tmpl,
                               const int vecLength) :
    EpiRK4SV(f,jtv,userData,maxKrylovIters,tmpl,vecLength)
{

}

// "ONESTEP" mode for the exponential integrator, I copy verbatim the source code from EPIC but remove the time loop
PacmenslErrorCode EpiRK4SVInterface::Step(const realtype hStart,
                                         const realtype hMax,
                                         const realtype absTol,
                                         const realtype relTol,
                                         const realtype t0,
                                         const realtype tFinal,
                                         const int numBands,
                                         int *basisSizes,
                                         N_Vector y,
                                         realtype *tnew,
                                         realtype *hnew)
{
  using namespace EpiRK4SVNamespace;
  if (hStart < ZERO)
  {
    printf("Initial time step h is to small. \n");
    return -1;
  }
  if (tFinal < t0)
  {
    printf("Starting time is larger the end time. \n");
    return -1;
  }
  realtype t    = t0;
  realtype hNew = hStart;
  realtype h    = hStart;
  if (hMax < hStart)
  {
    hNew = hMax;
    h    = hMax;
  }

  if (t0 + hNew >= tFinal)
  {
    hNew = tFinal - t0;
  }


  realtype krylovTol = 1.0e-14;

  realtype err = 5.0;
  while (err > 1)
  {
    h = hNew;

    // f is RHS function from the problem; y = u_n
    f(t, y, fy, userData);
    N_VScale(h, fy, hfy);
    JTimesV jtimesv(jtv, f, delta, t, y, fy, userData, tmpVec);

    // Stage 1.
    N_Vector stage1InputVecs[] = {zeroVec, hfy};
    N_Vector stage1OutputVecs[] = {r1, r2, r3LowOrder};
    krylov->Compute(numBands, Stage1NumInputVecs, stage1InputVecs, g1Times, g1NumTimes, stage1OutputVecs, &jtimesv, h, krylovTol, basisSizes[0], &integratorStats->krylovStats[0]);
    // computes phi_k(g_{i j} h J)*hfy and stores it in ri, ri is r(U_i)
    // in this case r1         = phi_1( 1/2 h J_n ) * hfy
    // and          r2         = phi_1( 2/3 h J_n ) * hfy
    // and          r3LowOrder = phi_1( 3/4 h J_n ) * hfy

    // States U_2 and U_3 are now being computed and results are stored in r1 and r2, respectively
    N_VLinearSum(1.0, y, a21, r1, r1); // r1 = y + a21 * r1 = y + a21 * phi( ) * hfy
    N_VLinearSum(1.0, y, a31, r2, r2); // r2 = y + a31 * r1 = y + a31 * phi( ) * hfy

    // Stage 3 - High-order part
    // In this part we compute residual of U_2, i.e., h*r(U_2). Result will be stored in hb1 variable
    N_VLinearSum(1.0, r1, -1.0, y, scratchVec1); // scratchVec1 = a21 * phi( ) * hfy
    jtimesv.ComputeJv(scratchVec1, scratchVec2); // scratchVec1 can be reused now, scratchVec2 is Jacobian of scratchVec1
    f(t, r1, hb1, userData); // f(t, r1) = hb1
    N_VLinearSum(h, hb1, -1.0, hfy, hb1); // hb1 = h * hb1 - hfy
    N_VLinearSum(1.0, hb1, -1.0*h, scratchVec2, hb1); // scratchVec2 can be reused now, hb1 contains residual of U_2
    // In order to compute residual we did the following
    // hb1 = hb1 - h*scratchVec2
    //     = h*hb1 - hfy - Jacobian(scratchVec1)
    //     = f(t, r1) - hfy - Jacobian(r1-y)
    //     = f(t, y + a21*phi( )*hfy) - hfy - Jacobian( a21*phi( )*hfy )*hfy
    //     = h*r(U_2) <- residual

    // scratchVec1 and scratchVec2 are free variables

    // In this part we compute residual of U_3, i.e., h*r(U_3). Result will be stored in hb2 variable
    // We preform same caluculation we did for residual of U_2
    N_VLinearSum(1.0, r2, -1.0, y, scratchVec1);
    jtimesv.ComputeJv(scratchVec1, scratchVec2);
    f(t, r2, hb2, userData);
    N_VLinearSum(h, hb2, -1.0, hfy, hb2);
    N_VLinearSum(1.0, hb2, -1.0*h, scratchVec2, hb2);
    // Similar like for hb1, hb2 is now h*r(U_3)

    // scratchVec1 and scratchVec2 are free variables

    // State u_{n+1} is now being computed
    N_VLinearSum(b2a, hb1, b2b, hb2, scratchVec1);
    N_VLinearSum(b3a, hb1, b3b, hb2, scratchVec2);

    // scratchVec1 and scratchVec2 contain linear combinations of residuals

    N_Vector stage3InputVecs[] = {zeroVec, hfy, zeroVec, scratchVec1, scratchVec2};
    krylov2->Compute(numBands, Stage3NumInputVecs, stage3InputVecs, scratchVec3, 1.0, &jtimesv, h, krylovTol, basisSizes[1], &integratorStats->krylovStats[1]);
    // scratchVec3 now holds high-order portion, i.e.,
    // scratchVec3 = phi_1(h J_n) h f(u_n) + (32 phi_3(h J_n) - 144 phi_4(h J_n)) h r(U_2)
    //                                     + (-27/2 phi_3(h J_n) + 81 phi_4 (h J_n)) h r(U_3)

    N_VLinearSum(1.0, y, 1.0, scratchVec3, scratchVec1);
    // scratchVec1 now holds tentative new y

    // Lower order
    // State U_2 for low order is being computed and results is stored in r3LowOrder
    N_VLinearSum(1.0, y, a21LowOrder, r3LowOrder, r3LowOrder); // r3LoweOrder = y + a21 * phi ( )*hfy

    // In this part we compute residual of U_2 for low order, i.e., h*r(U_2). Result will be stored in hb1 variable
    N_VLinearSum(1.0, r3LowOrder, -1.0, y, scratchVec5); // scratchVec5 = a21 * phi( ) * hfy
    jtimesv.ComputeJv(scratchVec5, scratchVec6);
    f(t, r3LowOrder, hb1, userData); // f(t, r3LowOrder) = hb1
    N_VLinearSum(h, hb1, -1.0, hfy, hb1); // hb1 = h * hb1 - hfy
    N_VLinearSum(1.0, hb1, -1.0*h, scratchVec6, hb1);// scratchVec6 can be reused now, hb1 contains residual of U_2
    // In order to compute residual we did the following
    // hb1 = hb1 - h*scratchVec6
    //     = h*hb1 - hfy - Jacobian(scratchVec5)
    //     = f(t, r3LowOrder) - hfy - Jacobian(r3LowOrder-y)
    //     = f(t, y + a21LowOrder*phi( )*hfy) - hfy - Jacobian( a21LowOrder*phi( )*hfy )*hfy
    //     = h*r(U_2) <- residual

    // State u_{n+1} low order is now being computed
    N_VScale(b2aLowOrder, hb1, scratchVec2);
    N_VScale(b2bLowOrder, hb1, scratchVec5);

    // scratchVec2 and scratchVec5 contain linear combinations of residuals

    N_Vector stage3LowOrderInputVecs[] = {zeroVec, hfy, zeroVec, scratchVec2, scratchVec5};
    krylov2->Compute(numBands, Stage3NumInputVecsLowOrder, stage3LowOrderInputVecs, scratchVec4, 1.0, &jtimesv, h, krylovTol, basisSizes[2], &integratorStats->krylovStats[2]);
    // scratchVec4 now holds low order portion, i.e.,
    // scratchVec4 = phi_1( h J_n) h f(u_n)
    //                + (p_{223} phi_3 (h J_n) + (128/9 - 4 p_{223}) phi_4(h J_n)) h r(U_2)

    // Estimate error.
    N_VLinearSum(1.0, scratchVec3, -1.0, scratchVec4, scratchVec4);
    N_VAbs(y, scratchVec3);
    N_VScale(relTol, scratchVec3, scratchVec3);
    N_VAddConst(scratchVec3, absTol, scratchVec3);
    N_VDiv(scratchVec4, scratchVec3, scratchVec4);  // can now re-use scratchVec3
    realtype norm = N_VDotProd(scratchVec4, scratchVec4);  // can now re-use scratchVec6
    norm = norm / NEQ;
    err = EPICRSqrt(norm);
    hNew = h * Fac * pow(err, Order);
    if (hNew > hMax)
    {
      hNew = hMax;
    }
    if (hNew < ZERO)
    {
      printf("There is possible singularity in the solution\n");
      exit(EXIT_FAILURE);
    }
    //printf("err = %f, hNew = %f\n", err, hNew);
  }

  if (t + hNew >= tFinal)
  {
    hNew = tFinal - t;
  }
  *hnew = hNew;
  // Copy tentative y (scratchVec1) to y.
  N_VLinearSum(1.0,scratchVec1,0.0,y,y);
  // y is now new y, i.e., u_{n+1} = y

  *tnew = t + h;

  return 0;
}
}