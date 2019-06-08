//
// Created by Huy Vo on 6/2/19.
//

#ifndef PFSPAT_FSPMATRIXCONSTRAINED_H
#define PFSPAT_FSPMATRIXCONSTRAINED_H

#include "FspMatrixBase.h"

namespace cme{
    namespace parallel{
        class FspMatrixConstrained: public cme::parallel::FspMatrixBase{
        protected:
            int num_constraints_;
            int sinks_rank_; ///< rank of the processor that stores sink states
            Mat sinks_mat_; ///< local matrix to evaluate sink states
            Vec sink_entries_;
            VecScatter sink_scatter_ctx_;

            virtual void determine_layout(const StateSetBase &fsp) override;
        public:
            explicit FspMatrixConstrained(MPI_Comm comm);

            void generate_matrices( cme::parallel::StateSetConstrained &fsp, const arma::Mat< int > &SM,
                                    cme::parallel::PropFun prop,
                                    cme::parallel::TcoefFun new_t_fun );

            void destroy( ) override;

            void action( PetscReal t, Vec x, Vec y ) override;

            ~FspMatrixConstrained( );
        };
    }
}

#endif //PFSPAT_FSPMATRIXCONSTRAINED_H
