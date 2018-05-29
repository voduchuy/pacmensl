#pragma once

#include <vector>
#include <armadillo>
#include <petscmat.h>
#include <petscvec.h>
#include "KExpv.h"

namespace cme {
    namespace petsc {
        /**
         * @brief Wrapper class for order 4 Magnus integration of the non-autonomous truncated CME. The integration is stopped
         * when the FSP criteria is violated.
         */
        class Magnus4FSP {
            using TMVFun = std::function<void(Real t, Vec x, Vec y)>;
            using Real = PetscReal;
            using Int = PetscInt;

        private:
            KExpv expv;

            static void magnus_mv(Vec x, Vec y, Magnus4FSP *magnus_ts);

            Real t_step_max;
            Real t_step;

            Int m_krylov;
            Real kry_tol;

            /* Work space */
            Int n_global;
            Vec w1, w2, w3, w4, w5, solution_old;
            const Real sqrt3 = 1.732050807568877;


            /**
             * @brief Method to get the values of sink states
             */
            void get_sinks();

        public:
            const MPI_Comm comm = MPI_COMM_NULL;       ///< Commmunicator for the time-stepping scheme. This must be the same as the one used for the vector and matrix involved.

            TMVFun tmatvec;
            Vec solution_now;

            Real t_start;       ///< Starting time
            Real t_final;       ///< Final time

            Real t_now;

            Int i_step = 0;

            arma::Row<Int> to_expand; ///< Logical array for whether a sink state needs expansion

            /* Adjustable algorithmic parameters */
            Real fsp_tol;
            Real tol = 1.0e-4;
            Int max_nstep = 10000;
            NormType local_error_norm = NORM_2;

            /* Armadillo vector to store sink states */
            arma::Row<Real> sinks;
            Int n_sinks;
            arma::Row<Int> sink_indices;
            Int sink_rank; ///< Rank of the process that contains the sink states

            /**
             * @brief Constructor without vector data structures
             */
             Magnus4FSP(MPI_Comm _comm, Real _t_start, Real _t_final, Int _n_sinks, Real _fsp_tol, Real _tol = 1.0e-8, Real _kry_tol = 1.0e-8,
                     Int _m = 30, bool _iop = true, Int _q_iop = 2,  Real _anorm = 1.0):
            comm(_comm),
            t_start(_t_start),
            t_final(_t_final),
            fsp_tol(_fsp_tol),
            tol(_tol),
            m_krylov(_m),
            kry_tol(_kry_tol),
            n_sinks(_n_sinks),
            expv( _comm,  _t_final,  _m,  _kry_tol, _iop, _q_iop, _anorm)
            {

            }
            /**
             * @brief Constructor for Magnus4FSP object.
             */
            Magnus4FSP(MPI_Comm _comm, Real _t_start, Real _t_final, TMVFun _tmatvec, Vec _v, Int _n_sinks,
                       Real _fsp_tol,
                       Real _tol = 1.0e-8,
                       Real _kry_tol = 1.0e-8,
                       Int _m = 30,
                       bool _iop = true, Int _q_iop = 2, Real _anorm = 1.0) :
                    comm(_comm),
                    t_start(_t_start),
                    t_final(_t_final),
                    tmatvec(_tmatvec),
                    solution_now(_v),
                    fsp_tol(_fsp_tol),
                    tol(_tol),
                    m_krylov(_m),
                    kry_tol(_kry_tol),
                    n_sinks(_n_sinks),
                    expv(_comm, _t_final, [this](Vec x, Vec y) { magnus_mv(x, y, this); }, _v, _m, _kry_tol) {
                t_now = t_start;
                update_vector(_v, _tmatvec);
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
            void destroy() {
                expv.destroy();
                VecDestroy(&w1);
                VecDestroy(&w2);
                VecDestroy(&w3);
                VecDestroy(&w4);
                VecDestroy(&w5);
                VecDestroy(&solution_old);
            }

            /* TODO: method to reset data structures when a new starting vector and matvec function is selected */
            void update_vector(Vec _v, TMVFun _tmatvec) {
                solution_now = _v;
                expv.update_vectors(_v, [this](Vec x, Vec y) { magnus_mv(x, y, this); });
                tmatvec = _tmatvec;
                t_step_max = t_final - t_now;
                sinks.set_size(n_sinks);
                to_expand.set_size(n_sinks);
                VecGetSize(solution_now, &n_global);
                sink_indices = arma::linspace<arma::Row<Int>>(n_global - n_sinks, n_global - 1, n_sinks);
                std::cout << sink_indices;

                Int itmp{0}, i1, i2;
                VecGetOwnershipRange(solution_now, &i1, &i2);
                if (i2 == n_global) {
                    MPI_Comm_rank(comm, &itmp);
                }
                MPI_Allreduce(&itmp, &sink_rank, 1, MPI_INT, MPI_MAX, comm);

                VecDuplicate(solution_now, &w1);
                VecDuplicate(solution_now, &w2);
                VecDuplicate(solution_now, &w3);
                VecDuplicate(solution_now, &w4);
                VecDuplicate(solution_now, &w5);
                VecDuplicate(solution_now, &solution_old);
            }
        };
    }
}
