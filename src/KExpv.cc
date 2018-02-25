#include "KExpv.hpp"

namespace cme {
namespace petsc {
void KExpv::step()
{
        Real beta, s, avnorm, xm, err_loc;

        VecNorm(solution_now, NORM_2, &beta);
        if (i_step == 0)
        {
                Real xm = 1.0/Real(m);
                //double anorm { norm(A, 1) };
                Real fact = pow( (m+1)/exp(1.0), m+1 )*sqrt( 2*(3.1416)*(m+1) );
                t_new = (1.0/anorm)*pow( (fact*tol)/(4.0*beta*anorm), xm);
                btol = anorm*tol; // tolerance for happy breakdown
        }

        Int mx;
        Int mb {m};
        Int k1 {2};

        PetscReal tau = std::min( t_final - t_now, t_new );
        H = arma::zeros( m+2, m+2 );

        VecCopy(solution_now, V[0]);
        VecScale(V[0], 1.0/beta);

        Int istart = 0;
        /* Arnoldi loop */
        for ( int j{0}; j < m; j++ )
        {
                matvec(V[j], V[j+1]);

                /* Orthogonalization */

                if (IOP) istart = (j - q_iop + 1 >= 0 ) ? j - q_iop + 1 : 0;
                for ( int i { istart }; i <= j; i++ )
                {
                        VecDot(V[j+1], V[i], &H(i,j));
                        VecAXPY(V[j+1], -1.0*H(i,j), V[i]);
                }

                VecNorm(V[j+1], NORM_2, &s);

                if ( s< btol )
                {
                        k1 = 0;
                        mb = j+1;
                        tau = t_final - t_now;
                        PetscPrintf(comm, "Happy breakdown.");
                        break;
                }

                H( j+1, j ) = s;
                VecScale(V[j+1], 1.0/s);

        }


        if ( k1 != 0 )
        {
                H( m+1, m ) = 1.0;
                matvec(V[mb], av);
                VecNorm( av, NORM_2, &avnorm );
        }

        int ireject {0};
        while ( ireject < max_reject )
        {
                mx = mb + k1;
                F = expmat(tau*H);
                //std::cout << F << std::endl;
                if ( k1 == 0 )
                {
                        err_loc = btol;
                        break;
                }
                else
                {
                        double phi1 = std::abs( beta*F( m, 0) );
                        double phi2 = std::abs( beta*F( m+1, 0)*avnorm );

                        if ( phi1 > phi2*10.0 )
                        {
                                err_loc = phi2;
                                xm = 1.0/double(m);
                        }
                        else if ( phi1 > phi2 )
                        {
                                err_loc = (phi1*phi2)/(phi1-phi2);
                                xm = 1.0/double(m);
                        }
                        else
                        {
                                err_loc = phi1;
                                xm = 1.0/double(m-1);
                        }
                }

                if ( err_loc <= delta*tau*tol )
                {
                        break;
                }
                else
                {
                        tau = gamma * tau * pow( tau*tol/err_loc, xm );
                        double s = pow( 10.0, floor(log10(tau)) - 1 );
                        tau = ceil(tau/s) * s;
                        if (ireject == max_reject)
                        {
                                // This part could be dangerous, what if one processor exits but the others continue
                                PetscPrintf(comm, "Maximum number of failed steps reached.");
                                t_now = t_final;
                                break;
                        }
                        ireject++;
                }
        }

        mx = mb + std::max( 0, k1-1 );
        arma::Col<double> F0 = beta*F( arma::span(0, mx-1), 0 );

        VecScale(solution_now, F0(0)/beta);
        for ( size_t i{1}; i < mx; i++)
        {
                VecAXPY(solution_now, F0(i), V[i]);
        }

        t_now = t_now + tau;
        t_new = gamma*tau*pow( tau*tol/err_loc, xm );
        s = pow( 10.0, floor(log10(t_new) ) - 1.0);
        t_new = ceil( t_new/s )*s;

#ifdef KEXPV_VERBOSE
        PetscPrintf(comm, "t_now = %2.2e \n", t_now);
#endif
        i_step++;

}

void KExpv::solve()
{
        while (!final_time_reached())
        {
                step();
        }
}

}
}
