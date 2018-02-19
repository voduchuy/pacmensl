#include "Magnus4.hpp"

namespace cme {
namespace petsc {
void Magnus4::magnus_mv(Vec x, Vec y, Magnus4* magnus_ts)
{
        Real& t_now = magnus_ts->t_now;
        Real& t_step = magnus_ts->t_step;
        const Real& sqrt3 = magnus_ts->sqrt3;
        Vec& w1 = magnus_ts->w1;
        Vec& w2 = magnus_ts->w2;
        Vec& w3 = magnus_ts->w3;
        Vec& w4 = magnus_ts->w4;

        Real t1 = t_now + (0.5 - sqrt3/6.0)*t_step;
        Real t2 = t_now + (0.5 + sqrt3/6.0)*t_step;
        magnus_ts->tmatvec(t1, x, w1);
        magnus_ts->tmatvec(t2, x, w2);
        magnus_ts->tmatvec(t2, w1, w3);
        magnus_ts->tmatvec(t1, w2, w4);

        VecAXPY(w1, 1.0, w2);
        VecAXPY(w3, -1.0, w4);

        VecScale(w1, 0.5*t_step);

        VecCopy(w1, y);
        VecAXPY(y, t_step*t_step*sqrt3/12.0, w3);
}

void Magnus4::step()
{
        if ( i_step == 0 ) t_step = 1.0;

        Real local_error;
        do {
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


                local_error = (t_step)*(t_step)*(t_step)*(t_step)*xtmp/720.0;

                if (local_error >= 1.2e0*t_step*tol) {
                        t_step = 0.9e0*pow( t_step*tol/local_error, 0.25e0)*t_step;
                }

        } while (local_error >= 1.2e0*t_step*tol);

        t_step = std::min(t_step, t_final - t_now);

        /* Advance to the next step with matrix exponential */
        expv.reset_time(t_step);
        expv.solve();

        /* Update time */
        t_now += t_step;
        i_step++;

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


        local_error = (t_step)*(t_step)*(t_step)*(t_step)*xtmp/720.0;

        t_step = 0.9e0*pow( t_step*tol/local_error, 0.25e0)*t_step;

        PetscPrintf(comm, "%d t = %2.4e dt = %2.4e \n", i_step, t_now, t_step);
}

void Magnus4::solve()
{
        while (t_now < t_final)
        {
                step();
        }
}
}
}
