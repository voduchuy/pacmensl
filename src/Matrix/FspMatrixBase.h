#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <armadillo>
#include <mpi.h>
#include <petscmat.h>
#include <petscis.h>
#include "util/cme_util.h"
#include "FSS/StateSetBase.h"

namespace cme {
    namespace parallel {
        using PropFun = std::function<PetscReal(PetscInt *, PetscInt)>;
        using TcoefFun = std::function<arma::Row<PetscReal>(PetscReal t)>;
        using Real = PetscReal;
        using Int = PetscInt;


        /// Distributed data type for the truncated CME operator on a hyper-rectangle
        /**
         *
         **/
        class FspMatrixBase {

        protected:
            MPI_Comm comm{MPI_COMM_NULL};

            arma::Row<int> fsp_bounds;

            Int n_reactions;
            Int n_rows_global;
            Int n_rows_local;

            // Local data of the matrix
            std::vector<Mat> diag_mats;
            std::vector<Mat> offdiag_mats;

            // Data for computing the matrix action
            Vec work; ///< Work vector for computing operator times vector
            Vec lvec; ///< Local vector to receive scattered data from the input vec
            PetscInt lvec_length; ///< Number of ghost entries owned by the local process
            VecScatter action_ctx; ///< Scatter context for computing matrix action
            Vec xx, yy, zz; ///< Local portion of the vectors

            TcoefFun t_fun = NULL;

        public:

            /* constructors */
            explicit FspMatrixBase(MPI_Comm _comm);

            void GenerateMatrices(StateSetBase &fsp, const arma::Mat<Int> &SM, PropFun prop, TcoefFun new_t_fun);

            void Destroy();

            void Action(PetscReal t, Vec x, Vec y);

            PetscInt GetLocalGhostLength();

            ~FspMatrixBase();
        };
    }
}
