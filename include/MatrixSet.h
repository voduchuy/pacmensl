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
#include <petscao.h>
#include "cme_util.h"
#include "FiniteStateSubset.h"
#include "FiniteStateSubsetGraph.h"
#include "FiniteStateSubsetNaive.h"

namespace cme {
    namespace petsc {
        using PropFun = std::function<PetscReal(PetscInt *, PetscInt)>;
        using TcoefFun = std::function<arma::Row<PetscReal>(PetscReal t)>;
        using Real = PetscReal;
        using Int = PetscInt;

        /* Distributed data type for the truncated CME operator on a hyper-rectangle */
        class MatrixSet {

        protected:
            MPI_Comm comm{MPI_COMM_NULL};

            arma::Row<Int> fsp_size;

            Int n_reactions;
            Int n_rows_global;
            Int n_rows_local;
            std::vector<Mat> terms;

            Vec work; ///< Work vector for computing operator times vector
            TcoefFun t_fun = NULL;

        public:

            /* constructors */
            explicit MatrixSet(MPI_Comm _comm);

            void GenerateMatrices(FiniteStateSubset &fsp, const arma::Mat<Int> &SM, PropFun prop, TcoefFun new_t_fun);

            void Destroy();

            void DuplicateStructure(Mat &A);

            void PrintInfo();

            void Action(PetscReal t, Vec x, Vec y);

            ~MatrixSet();
        };
    }
}
