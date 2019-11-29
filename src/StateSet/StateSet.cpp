//
// Created by Huy Vo on 2019-06-18.
//

#include "StateSet.h"

pacmensl::StateSet::StateSet( MPI_Comm comm, pacmensl::StateSetType type, int num_species,
                              pacmensl::PartitioningType part, pacmensl::PartitioningApproach repart ) {
    type_ = type;
    switch ( type ) {
        case BASE:
            state_set_ = new StateSetBase(comm);
            break;
        case CONSTRAINED:
            state_set_ = new StateSetConstrained(comm);
            break;
    }
}

void pacmensl::StateSet::Expand( ) {
    state_set_->Expand();
}

pacmensl::StateSet::~StateSet( ) {
    switch ( type_ ) {
        case BASE:
            delete state_set_;
            break;
        case CONSTRAINED:
            delete ((StateSetConstrained*) state_set_);
            break;
    }
}
