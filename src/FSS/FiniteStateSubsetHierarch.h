//
// Created by Huy Vo on 12/4/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETHIERARCH_H
#define PARALLEL_FSP_FINITESTATESUBSETHIERARCH_H

#include "FiniteStateSubset.h"

namespace cme {
    namespace parallel {
        class FiniteStateSubsetHierarch : public FiniteStateSubset {
            static const int num_levels = 2; // Partitioning on the inter-node and intra-node levels
            int my_part[num_levels]; // The partition number of this processor for each level

            MPI_Comm my_node;

            PetscBool repart = PETSC_FALSE;

            void set_zoltan_parameters(int level, Zoltan_Struct *zz);

            friend int zoltan_hier_num_levels (void *data, int *ierr);

            friend int zoltan_hier_part (void *data, int level, int *ierr);

            friend void zoltan_hier_method (void *data, int level, struct Zoltan_Struct * zz, int *ierr);


        public:
            explicit FiniteStateSubsetHierarch(MPI_Comm new_comm);

            void GenerateStatesAndOrdering() override;

            void ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) override;

            ~FiniteStateSubsetHierarch();
        };

        int zoltan_hier_num_levels (void *data, int *ierr);

        int zoltan_hier_part (void *data, int level, int *ierr);

        void zoltan_hier_method (void *data, int level, struct Zoltan_Struct * zz, int *ierr);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETHIERARCH_H
