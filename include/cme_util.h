#pragma once

#include <armadillo>
#include <petscmat.h>
#include <petscvec.h>
#include <petscao.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscoptions.h>
#include <petsc.h>
#include <petscsys.h>
#include <petscconf.h>
#include <cassert>

namespace cme {
/*
   The following functions mimic the similar MATLAB functions. Armadillo only supports up to 3 dimensions, thus the neccesity of writing our own code.
 */
    template<typename intT>
    arma::Row<intT> sub2ind_nd(const arma::Row<intT> &nmax, const arma::Mat<intT> &X) {
        arma::uword N = nmax.n_elem;
        arma::uword nst = X.n_cols;

        arma::Row<intT> indx(nst);

        intT nprod{1};
        for (size_t j{0}; j < nst; j++) {
            nprod = 1;
            indx(j) = 0;
            for (size_t i{0}; i < N; i++) {
                if (X(i, j) < 0) {
                    indx(j) = (intT) -1;
                    break;
                }

                if (X(i, j) > nmax(i)) {
                    indx(j) = (intT) -(i + 2);
                    break;
                }
                indx(j) += X(i, j) * nprod;
                nprod *= (nmax(i) + 1);
            }
        }

        return indx;
    };

    template<typename IntMatT, typename IntVecT1, typename IntVecT2>
    IntMatT ind2sub_nd(const IntVecT1 &nmax, const IntVecT2 &indx) {
        auto N = nmax.size();
        auto nst = indx.size();

        IntMatT X(N, nst);

        int k;
        for (auto j{1}; j <= nst; j++) {
            k = indx.at(j - 1);
            for (auto i{1}; i <= N; i++) {
                X(i - 1, j - 1) = k % (nmax(i - 1) + 1);
                k = k / (nmax(i - 1) + 1);
            }
        }

        return X;
    };
    template arma::Mat<PetscInt> ind2sub_nd<arma::Mat<PetscInt>, arma::Row<PetscInt>, arma::Row<PetscInt>>(const arma::Row<PetscInt> &nmax, const arma::Row<PetscInt> &indx);
    template arma::Mat<PetscInt> ind2sub_nd<arma::Mat<PetscInt>, arma::Row<PetscInt>, std::vector<PetscInt>>(const arma::Row<PetscInt> &nmax, const std::vector<PetscInt> &indx);

    template<typename intT>
    arma::Col<intT> ind2sub_nd(const arma::Row<intT> &nmax, const intT &indx) {
        arma::uword N = nmax.size();

        arma::Col<intT> X(N);

        int k;

        k = indx;
        for (size_t i{1}; i <= N; i++) {
            X(i - 1) = k % (nmax(i - 1) + 1);
            k = k / (nmax(i - 1) + 1);
        }

        return X;
    };

/*! Distribute a set of identical tasks as evenly as possible to a number of processors.
*/
    template<typename intT>
    arma::Row<intT> distribute_tasks(intT n_tasks, intT n_procs) {
        arma::Row<intT> dist(n_procs);
        dist.fill(0);

        intT i1 = 0;
        while (n_tasks > 0) {
            dist(i1) += 1;
            n_tasks -= 1;
            i1 += 1;
            if (i1 >= n_procs) i1 = 0;
        }

        return dist;
    }

/*! Find the range of task ids a process owns, assuming that each process own a contiguous range of tasks.
*/
    template<typename intT>
    std::pair<intT, intT> get_task_range(arma::Row<intT> job_dist, intT rank) {
        if (rank >= job_dist.n_elem) {
            throw "get_task_range: Requesting task range for a process out of range.\n";
        }

        intT i1, i2;
        i1 = 0;
        for (intT i = {1}; i < rank; ++i) {
            i1 += job_dist(i - 1);
        }
        i2 = i1 + job_dist(rank);
        return std::make_pair(i1, i2);
    }

}
