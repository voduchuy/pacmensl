//
// Created by Huy Vo on 6/2/19.
//

#ifndef PACMENSL_FSPMATRIXCONSTRAINED_H
#define PACMENSL_FSPMATRIXCONSTRAINED_H

#include "FspMatrixBase.h"

namespace pacmensl {
    class FspMatrixConstrained : public FspMatrixBase {
    protected:
        int num_constraints_;
        int sinks_rank_; ///< rank of the processor that stores sink states
        std::vector<Mat> sinks_mat_; ///< local matrix to evaluate sink states
        Vec sink_entries_, sink_tmp;
        VecScatter sink_scatter_ctx_;

        int DetermineLayout_( const StateSetBase &fsp) override;

    public:
        explicit FspMatrixConstrained(MPI_Comm comm);

        int GenerateValues( const StateSetBase &fsp, const arma::Mat< int > &SM, const PropFun &prop,
                            void *prop_args, const TcoefFun &new_t_fun, void *t_fun_args) override;

        int Destroy() override;

        int action( PetscReal t, Vec x, Vec y) override;

        ~FspMatrixConstrained();
    };
}

#endif //PACMENSL_FSPMATRIXCONSTRAINED_H
