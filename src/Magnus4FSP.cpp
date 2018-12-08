#include "Magnus4FSP.h"

namespace cme {
    namespace petsc {
        void Magnus4FSP::magnus_mv(Vec x, Vec y, Magnus4FSP *magnus_ts) {
            Real &t_now = magnus_ts->t_now;
            Real &t_step = magnus_ts->t_step;
            const Real &sqrt3 = magnus_ts->sqrt3;
            Vec &w1 = magnus_ts->w1;
            Vec &w2 = magnus_ts->w2;
            Vec &w3 = magnus_ts->w3;
            Vec &w4 = magnus_ts->w4;

            Real t1 = t_now + (0.5 - sqrt3 / 6.0) * t_step;
            Real t2 = t_now + (0.5 + sqrt3 / 6.0) * t_step;
            magnus_ts->tmatvec(t1, x, w1);
            magnus_ts->tmatvec(t2, x, w2);
            magnus_ts->tmatvec(t2, w1, w3);
            magnus_ts->tmatvec(t1, w2, w4);

            VecAXPY(w1, 1.0, w2);
            VecAXPY(w3, -1.0, w4);

            VecScale(w1, 0.5 * t_step);

            VecCopy(w1, y);
            VecAXPY(y, t_step * t_step * sqrt3 / 12.0, w3);
        }

        void Magnus4FSP::step() {
            if (i_step == 0) t_step = 1.0e-4;

            PetscReal t_step_new;
            PetscReal local_error;
            do {
                t_step = std::min(t_step, t_step_max);
                /* Compute the stepsize */
                // First vector
                tmatvec(t_now + t_step, solution_now, w5);
                tmatvec(t_now, w5, w1);
                tmatvec(t_now + t_step, w1, w5);
                tmatvec(t_now + t_step, w5, w1);
                // Second vector
                tmatvec(t_now + t_step, solution_now, w5);
                tmatvec(t_now + t_step, w5, w2);
                tmatvec(t_now, w2, w5);
                tmatvec(t_now + t_step, w5, w2);
                // Third vector
                tmatvec(t_now, solution_now, w5);
                tmatvec(t_now + t_step, w5, w3);
                tmatvec(t_now + t_step, w3, w5);
                tmatvec(t_now + t_step, w5, w3);
                // Fourth vector
                tmatvec(t_now + t_step, solution_now, w5);
                tmatvec(t_now + t_step, w5, w4);
                tmatvec(t_now + t_step, w4, w5);
                tmatvec(t_now, w5, w4);

                VecAXPY(w4, -1.0, w3);
                VecAXPY(w4, -3.0, w2);
                VecAXPY(w4, 3.0, w1);

                Real xtmp;
                VecNorm(w4, local_error_norm, &xtmp);


                local_error = (t_step) * (t_step) * (t_step) * (t_step) * xtmp / 720.0;
                PetscReal t_step_old = t_step;
                t_step_new = pow(0.5e0 *  tol / local_error, 0.25e0) * t_step;
                if (local_error >= 1.2e0 * tol) {

                    t_step = std::min(5 * t_step, std::max(t_step_new, 0.5 * t_step));
                }

            } while (local_error >= 1.2e0 * tol);

#ifdef MAGNUS4_VERBOSE
            PetscPrintf(comm, "%d t = %2.4e dt = %2.4e \n", i_step, t_now, t_step);
#endif
            /* Advance to the next step with matrix exponential */
            expv.reset_time(1.0);
            expv.solve();

            /* Update time */
            t_now += t_step;
            i_step++;

            t_step = std::min(std::min(5 * t_step, std::max(t_step_new, 0.5 * t_step)), t_step_max);
        }

        void Magnus4FSP::solve() {
            Real prob_sum{0.0};
            Real t_old;

            while (t_now < t_final) {
                t_old = t_now;
                VecCopy(solution_now, solution_old);

                t_step_max = std::min(t_step_max, t_final - t_now);

                step();

                /* Ensure FSPSolver criteria, if fails quit */
                get_sinks();
                Real err_bound = (fsp_tol/(Real) n_sinks) * std::pow( t_now/ t_final, 2.0);
                if ( sinks.max() >= err_bound) {
                    /* Set solution to the previous time point */
                    VecCopy(solution_old, solution_now);
                    t_now = t_old;

                    /* Determine which sink local_states need expansion */
                    to_expand.fill(0);
                    to_expand.elem( arma::find(sinks >= err_bound ) ).ones();

#ifdef MAGNUS4_VERBOSE
                    PetscPrintf(comm, "FSPSolver criteria failed. Trying with a bigger FSPSolver...\n");
#endif
                    break;
                }
            }
        }

        void Magnus4FSP::get_sinks() {
            Int i1, i2;
            VecGetOwnershipRange(solution_now, &i1, &i2);
            if (i2 == n_global) {
                VecGetValues(solution_now, n_sinks, sink_indices.begin(), sinks.begin());
            }
            MPI_Bcast(sinks.begin(), n_sinks, MPI_DOUBLE, sink_rank, comm);
        }
    }
}
