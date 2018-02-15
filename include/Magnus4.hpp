#pragma once
#include <vector>
#include <armadillo>
#include <petscmat.h>
#include <petscvec.h>
#include "KExpv.hpp"

namespace cme{
  namespace petsc{

    using TMVFun = std::function<void (Real t, Vec x, Vec y)>;
    /**
     * @brief Wrapper class for order 4 Magnus integration of the non-autonomous linear system of ODE.
     */
    class Magnus4{
    protected:
      const MPI_Comm comm = MPI_COMM_NULL;       ///< Commmunicator for the time-stepping scheme. This must be the same as the one used for the vector and matrix involved.

      TMVFun tmatvec;
      Vec solution_now;

      Real t_final;       ///< Final time/scaling of the matrix.

      Real t_step;
      Real t_now;

      Int i_step = 0;
    private:
      KExpv expv;

      static void magnus_mv(Vec x, Vec y, Magnus4* magnus_ts);

      /* Work space */
      Vec w1, w2, w3, w4, w5;
      const Real sqrt3 = 1.732050807568877;
    public:
      /* Adjustable algorithmic parameters */
      Real tol = 1.0e-4;
      Int max_nstep = 10000;
      NormType local_error_norm = NORM_2;

      /* Constructor */
      Magnus4(MPI_Comm _comm, Real _t_final, TMVFun _tmatvec, Vec _v, Real _tol = 1.0e-4, Int _m = 30, bool _iop = false, Int _q_iop = 2, Real _kry_tol = 1.0e-8, Real _anorm = 1.0):
      comm(_comm),
      t_final(_t_final),
      tmatvec(_tmatvec),
      solution_now(_v),
      tol(_tol),
      expv(comm, t_final, [this] (Vec x, Vec y) {magnus_mv(x, y, this);}, solution_now, _m, _kry_tol )
      {
        t_now = 0.0;

        VecDuplicate(solution_now, &w1);
        VecDuplicate(solution_now, &w2);
        VecDuplicate(solution_now, &w3);
        VecDuplicate(solution_now, &w4);
        VecDuplicate(solution_now, &w5);
      }

      /**
       * @brief Integrate all the way to t_final.
       */
      void solve();

      /**
       * @brief Advance by one step, the step size is chosen adaptively.
       */
      void step();

      /**
      * @brief Member function to destroy the time-stepper. Needs to call this manually when the object is no longer needed.
      */
      void destroy()
      {
        expv.destroy();
        VecDestroy(&w1);
        VecDestroy(&w2);
        VecDestroy(&w3);
        VecDestroy(&w4);
        VecDestroy(&w5);
      }

    };
  }
}
