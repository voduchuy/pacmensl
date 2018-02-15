#pragma once
#include <armadillo>
#include <petscmat.h>
#include <petscvec.h>

namespace cme
{
/*
   The following functions mimic the similar MATLAB functions. Armadillo only supports up to 3 dimensions, thus the neccesity of writing our own code.
 */
template<typename intT>
arma::Row<intT> sub2ind_nd( arma::Row<intT> &nmax, arma::Mat<intT> &X )
{
        arma::uword N = nmax.size();
        arma::uword nst = X.n_cols;

        arma::Row<intT> indx(nst);

        int nprod{1};
        for (size_t j {1}; j <= nst; j++)
        {
                nprod = 1;
                indx(j-1) = 0;
                for ( size_t i {1}; i<= N; i++)
                {
                        if (X(i-1, j-1) < 0 || X(i-1, j-1) > nmax(i-1))
                        {
                                indx(j-1) = -1;
                                break;
                        }
                        indx(j-1) += X(i-1, j-1)*nprod;
                        nprod *= ( nmax(i-1) + 1);
                }
        }

        return indx;
};

template< typename intT >
arma::Mat<intT> ind2sub_nd( arma::Row<intT> &nmax, arma::Row<intT> &indx )
{
        arma::uword N = nmax.size();
        arma::uword nst = indx.size();

        arma::Mat<intT> X(N, nst);

        int k;
        for (size_t j {1}; j <= nst; j++)
        {
                k = indx(j-1);
                for (size_t i {1}; i <= N; i++)
                {
                        X(i-1, j-1) = k%(nmax(i-1) + 1);
                        k = k/(nmax(i-1) + 1);
                }
        }

        return X;
};

namespace petsc
{
  arma::Col<PetscReal> marginal(Vec P, arma::Row<PetscInt> &nmax, PetscInt species);
}

}
