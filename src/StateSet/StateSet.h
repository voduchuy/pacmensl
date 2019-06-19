//
// Created by Huy Vo on 2019-06-18.
//

#ifndef PACMENSL_STATESET_H
#define PACMENSL_STATESET_H

#include "StateSetBase.h"
#include "StateSetConstrained.h"

namespace pacmensl{
    typedef enum StateSetType{BASE, CONSTRAINED};
    class StateSet {
    public:
        StateSet(MPI_Comm comm, StateSetType type, int num_species, PartitioningType part, PartitioningApproach repart);
        void Expand();
        ~StateSet();
    protected:
        StateSetType type_ = BASE;
        StateSetBase* state_set_;
    };
}


#endif //PACMENSL_STATESET_H
