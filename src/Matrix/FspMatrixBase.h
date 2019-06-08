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
#include "Model.h"
#include "util/cme_util.h"
#include "FSS/StateSetBase.h"
#include "FSS/StateSetConstrained.h"

namespace cme {
    namespace parallel {
        using Real = PetscReal;
        using Int = PetscInt;


        /// Distributed data type for the truncated CME operator on a hyper-rectangle
        /**
         *
         **/
        class FspMatrixBase {

        protected:
            MPI_Comm comm_{MPI_COMM_NULL};
            int my_rank_;
            int comm_size_;

            Int n_reactions_;
            Int n_rows_global_;
            Int n_rows_local_;

            // Local data of the matrix
            std::vector< Mat > diag_mats_;
            std::vector< Mat > offdiag_mats_;

            // Data for computing the matrix action
            Vec work_; ///< Work vector for computing operator times vector
            Vec lvec_; ///< Local vector to receive scattered data from the input vec
            PetscInt lvec_length_; ///< Number of ghost entries owned by the local process
            VecScatter action_ctx_; ///< Scatter context for computing matrix action
            Vec xx, yy, zz; ///< Local portion of the vectors

            TcoefFun t_fun_ = nullptr;

            virtual void determine_layout(const StateSetBase &fsp);
        public:
            NOT_COPYABLE_NOT_MOVABLE(FspMatrixBase);

            /* constructors */
            explicit FspMatrixBase( MPI_Comm comm );

            virtual void
            generate_values( const StateSetBase &fsp, const arma::Mat< Int > &SM, PropFun prop, TcoefFun new_t_fun );

            virtual void destroy( );

            virtual void action( PetscReal t, Vec x, Vec y );

            PetscInt get_local_ghost_length( ) const;

            int get_num_rows_local() const {return n_rows_local_;};

            ~FspMatrixBase( );
        };

    }
}
