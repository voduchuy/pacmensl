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
#include <mpi.h>
#include <zoltan.h>
#include <parmetis.h>

#define ZOLTANCHKERRABORT(comm, ierr){\
            if (ierr == ZOLTAN_FATAL){\
                PetscPrintf(comm, "Zoltan Fatal in %s at line %d\n", __FILE__, __LINE__);\
                MPI_Abort(comm, -1);\
            }\
}\

#define MPICHKERRABORT(comm, ierr){\
            if (ierr != 0){\
                PetscPrintf(comm, "MPI error in %s at line %d\n", __FILE__, __LINE__);\
                MPI_Abort(comm, -1);\
            }\
}\

#define NOT_COPYABLE_NOT_MOVABLE(object)\
            object( const object & ) = delete;\
            object &operator=( const object & ) = delete;\
\

namespace pecmeal {
#define SQR1 sqrt(0.1e0)

/*
   The following functions mimic the similar MATLAB functions.
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

    template<typename intT, typename intT_out>
    void sub2ind_nd(const arma::Row<intT> &nmax, const arma::Mat<intT> &X, intT_out *indx) {
        arma::uword N = nmax.n_elem;
        arma::uword nst = X.n_cols;

        intT nprod{1};
        for (size_t j{0}; j < nst; j++) {
            nprod = 1;
            indx[j] = intT_out(0);
            for (size_t i{0}; i < N; i++) {
                if (X(i, j) < 0) {
                    indx[j] = (intT_out) -1;
                    break;
                }

                if (X(i, j) > nmax(i)) {
                    indx[j] = (intT_out) -(i + 2);
                    break;
                }
                indx[j] += (intT_out) X(i, j) * nprod;
                nprod *= (nmax(i) + 1);
            }
        }
    };

    template<typename intT, typename intT_out>
    void sub2ind_nd(intT num_dimensions, intT *nmax, intT num_states, intT *states, intT_out *indx) {
        intT nprod{1};
        for (size_t j{0}; j < num_states; j++) {
            nprod = 1;
            indx[j] = intT_out(0);
            for (size_t i{0}; i < num_dimensions; i++) {
                if (states[i + j * num_dimensions] < 0) {
                    indx[j] = (intT_out) -1;
                    break;
                }

                if (states[i + j * num_dimensions] > nmax[i]) {
                    indx[j] = (intT_out) -(i + 2);
                    break;
                }
                indx[j] += (intT_out) states[i + j * num_dimensions] * nprod;
                nprod *= (nmax[i] + 1);
            }
        }
    }


    template<typename intT, typename intT_out>
    void ind2sub_nd(intT num_dimensions, intT *nmax, intT num_indxs, intT_out *indx, intT *states) {
        int k;
        for (auto j{1}; j <= num_indxs; j++) {
            k = indx[j - 1];
            for (auto i{1}; i <= num_dimensions; i++) {
                states[(j - 1) * num_dimensions + i - 1] = k % (nmax[i - 1] + 1);
                k = k / (nmax[i - 1] + 1);
            }
        }
    }

    template<typename IntVecT1 = arma::Row<PetscInt>, typename IntVecT2 = arma::Row<PetscInt>, typename IntMatT = arma::Mat<PetscInt>>
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

    template arma::Mat<PetscInt>
    ind2sub_nd<arma::Row<PetscInt>, arma::Row<PetscInt>, arma::Mat<PetscInt>>(const arma::Row<PetscInt> &nmax,
                                                                              const arma::Row<PetscInt> &indx);

    template arma::Mat<PetscInt>
    ind2sub_nd<arma::Row<PetscInt>, std::vector<PetscInt>, arma::Mat<PetscInt>>(
            const arma::Row<PetscInt> &nmax,
            const std::vector<PetscInt> &indx);

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
        intT i1, i2;
        i1 = 0;
        for (intT i = {1}; i < rank; ++i) {
            i1 += job_dist(i - 1);
        }
        i2 = i1 + job_dist(rank);
        return std::make_pair(i1, i2);
    }

/*! Find unique columns
 * The type must be an integer for the method to work properly.
 */
    template<typename intT>
    arma::Mat<intT>
    unique_columns(arma::Mat<intT>
                   X) {
        if (X.n_cols == 0) {
            return X;
        }
        arma::Row<intT> max_entries = arma::max(X, 1).t();
        arma::Row<intT> ids = sub2ind_nd(max_entries, X);
        arma::uvec(unique_colids) = arma::find_unique(ids);
        return X.cols(unique_colids);
    }

    /*! Round to 2 significant digits
     */
    double round2digit(double x);

/*! Initialize and finalize Parallel context
 *
 */
    int ParaFSP_init(int *argc, char ***argv, const char *help);

    int ParaFSP_finalize();

    void sequential_action(MPI_Comm comm, std::function<void(void *)> action, void *data);
}
